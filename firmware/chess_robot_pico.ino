/*
 * Chess Gantry Robot — Raspberry Pi Pico (RP2040) Firmware v2.1
 * =============================================================
 *
 * HARDWARE SAFETY:
 *   Electromagnet MUST use IRLZ44N N-MOSFET with:
 *     - 220Ω gate resistor (GPIO → gate)
 *     - 10kΩ pulldown (gate → source, ensures default OFF)
 *     - 1N4007 flyback diode across coil (cathode to +V, anode to drain)
 *   NEVER drive coil directly from GPIO. Omitting flyback diode WILL
 *   destroy the MOSFET from back-EMF.
 *
 * Pin Map (RP2040 — GP0/GP1 reserved for USB, no UART0 conflict):
 *   X Stepper: STEP=GP2, DIR=GP3, ENABLE=GP4
 *   Y Stepper: STEP=GP5, DIR=GP6, ENABLE=GP7
 *   Z Stepper: STEP=GP8, DIR=GP9, ENABLE=GP12
 *   Magnet MOSFET gate: GP10 (active HIGH)
 *   Relay (motor power kill): GP11 (active HIGH = power ON)
 *   MS1/MS2/MS3: GP13/GP14/GP15 (A4988: H/H/L = 1/16 µstep)
 *   Buttons: STOP=GP21, RESET=GP22
 *   Endstops: X=GP26, Y=GP27, Z=GP28 (NC, active LOW with pullup)
 *   LED: GP25 (onboard)
 *
 * Camera connects to HOST, NOT Pico. Buck converter has no data pin.
 *
 * Dependencies: AccelStepper (PlatformIO / Arduino Library Manager)
 * Board core: arduino-pico (Earle Philhower)
 */

#include <AccelStepper.h>

// ===================== PIN DEFINITIONS =====================
#define X_STEP  2
#define X_DIR   3
#define X_EN    4
#define Y_STEP  5
#define Y_DIR   6
#define Y_EN    7
#define Z_STEP  8
#define Z_DIR   9
#define Z_EN    12

#define MAG_PIN    10   // IRLZ44N gate
#define RELAY_PIN  11   // Motor power relay
#define MS1_PIN    13
#define MS2_PIN    14
#define MS3_PIN    15

#define BTN_STOP   21
#define BTN_RESET  22
#define LED_PIN    25

#define X_END  26   // NC endstop, pulled HIGH, LOW = triggered
#define Y_END  27
#define Z_END  28

// ===================== MOTION PARAMS =====================
// NEMA17 1.8° = 200 steps/rev, 1/16 µstep = 3200/rev
// GT2 20T pulley = 40mm/rev → 80 µsteps/mm
#define STEPS_MM_XY  80.0f
#define STEPS_MM_Z   80.0f

#define X_MAX_MM  300.0f
#define Y_MAX_MM  300.0f
#define Z_MAX_MM   60.0f

#define XY_SPEED   100.0f  // mm/s
#define Z_SPEED     30.0f
#define XY_ACCEL   200.0f  // mm/s²
#define Z_ACCEL    100.0f

#define HOME_FAST    20.0f  // mm/s first pass
#define HOME_SLOW     1.0f  // mm/s re-approach
#define HOME_BACKOFF  5.0f  // mm backoff

#define WDT_MS     2000
#define MOTION_WDT_MULT 1.5f

// ===================== STATE MACHINE =====================
enum State {
    S_BOOT, S_NEED_HOME,
    S_HOME_Z1, S_HOME_Z_BACK, S_HOME_Z2,
    S_HOME_X1, S_HOME_X_BACK, S_HOME_X2,
    S_HOME_Y1, S_HOME_Y_BACK, S_HOME_Y2,
    S_HOME_FINAL,
    S_IDLE, S_MOVING, S_HALT, S_ERROR
};

// ===================== GLOBALS =====================
AccelStepper sX(AccelStepper::DRIVER, X_STEP, X_DIR);
AccelStepper sY(AccelStepper::DRIVER, Y_STEP, Y_DIR);
AccelStepper sZ(AccelStepper::DRIVER, Z_STEP, Z_DIR);

State state = S_BOOT;
bool magOn = false, relayOn = false, homed = false;
unsigned long lastCmdMs = 0, lastHbMs = 0;
unsigned long moveStartMs = 0, moveTimeoutMs = 0;
uint16_t moveSeq = 0;

#define BUF 128
char buf[BUF];
int bi = 0;
bool btnStopPrev = false, btnResetPrev = false;

// ===================== FORWARD DECLS =====================
void processCmd(const char* c);
void ack(uint16_t s, const char* st, const char* d = "");
void evt(const char* e, const char* d = "");
void halt(const char* r);
void setMag(bool on);
void setRelay(bool on);
bool endstop(int pin);
void enableMotors(bool en);
void startHome();

// ===================== SETUP =====================
void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000);

    pinMode(LED_PIN, OUTPUT);
    pinMode(MAG_PIN, OUTPUT); digitalWrite(MAG_PIN, LOW);
    pinMode(RELAY_PIN, OUTPUT); digitalWrite(RELAY_PIN, HIGH); relayOn = true;
    pinMode(X_EN, OUTPUT); pinMode(Y_EN, OUTPUT); pinMode(Z_EN, OUTPUT);
    enableMotors(true);

    pinMode(X_END, INPUT_PULLUP);
    pinMode(Y_END, INPUT_PULLUP);
    pinMode(Z_END, INPUT_PULLUP);
    pinMode(BTN_STOP, INPUT_PULLUP);
    pinMode(BTN_RESET, INPUT_PULLUP);

    // A4988 1/16: MS1=H MS2=H MS3=L
    pinMode(MS1_PIN, OUTPUT); digitalWrite(MS1_PIN, HIGH);
    pinMode(MS2_PIN, OUTPUT); digitalWrite(MS2_PIN, HIGH);
    pinMode(MS3_PIN, OUTPUT); digitalWrite(MS3_PIN, LOW);

    sX.setMaxSpeed(XY_SPEED * STEPS_MM_XY);
    sX.setAcceleration(XY_ACCEL * STEPS_MM_XY);
    sY.setMaxSpeed(XY_SPEED * STEPS_MM_XY);
    sY.setAcceleration(XY_ACCEL * STEPS_MM_XY);
    sZ.setMaxSpeed(Z_SPEED * STEPS_MM_Z);
    sZ.setAcceleration(Z_ACCEL * STEPS_MM_Z);

    state = S_NEED_HOME;
    lastCmdMs = millis();
    evt("BOOT", "v2.1 stepper gantry");
}

// ===================== MAIN LOOP =====================
void loop() {
    unsigned long now = millis();

    // Serial input (non-blocking)
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (bi > 0) { buf[bi] = 0; processCmd(buf); bi = 0; lastCmdMs = now; }
        } else if (bi < BUF - 1) buf[bi++] = c;
    }

    // Buttons (debounced edge detection)
    bool stop = !digitalRead(BTN_STOP);
    bool reset = !digitalRead(BTN_RESET);
    if (stop && !btnStopPrev) { halt("BTN_STOP"); evt("BTN_STOP"); }
    if (reset && !btnResetPrev && (state == S_HALT || state == S_ERROR)) {
        setMag(false); homed = false; state = S_NEED_HOME;
        enableMotors(true); setRelay(true);
        evt("BTN_RESET");
    }
    btnStopPrev = stop; btnResetPrev = reset;

    // Watchdog
    if ((state == S_IDLE || state == S_MOVING) && (now - lastCmdMs) > WDT_MS) {
        halt("WDT_TIMEOUT");
        setRelay(false);
    }

    // Per-motion watchdog
    if (state == S_MOVING && moveTimeoutMs > 0 && (now - moveStartMs) > moveTimeoutMs) {
        halt("MOTION_TIMEOUT");
    }

    // State machine
    switch (state) {
        case S_BOOT: case S_NEED_HOME:
            digitalWrite(LED_PIN, (now / 500) % 2); break;
        case S_HOME_Z1: case S_HOME_Z_BACK: case S_HOME_Z2:
        case S_HOME_X1: case S_HOME_X_BACK: case S_HOME_X2:
        case S_HOME_Y1: case S_HOME_Y_BACK: case S_HOME_Y2:
        case S_HOME_FINAL:
            handleHome(); break;
        case S_IDLE:
            digitalWrite(LED_PIN, HIGH); break;
        case S_MOVING:
            handleMove(); break;
        case S_HALT:
            digitalWrite(LED_PIN, (now / 100) % 2); setMag(false); break;
        case S_ERROR:
            digitalWrite(LED_PIN, (now / 50) % 2); setMag(false); break;
    }

    sX.run(); sY.run(); sZ.run();

    // Position heartbeat every 500ms
    if (state >= S_IDLE && state <= S_MOVING && (now - lastHbMs) > 500) {
        lastHbMs = now;
        char d[80];
        snprintf(d, sizeof(d), "X%.2f Y%.2f Z%.2f",
            sX.currentPosition() / STEPS_MM_XY,
            sY.currentPosition() / STEPS_MM_XY,
            sZ.currentPosition() / STEPS_MM_Z);
        evt("POS", d);
    }

    // Endstop safety during non-homing motion
    if (state == S_MOVING) {
        if (endstop(X_END)) { halt("ENDSTOP_X"); evt("EVT:ENDSTOP_X"); return; }
        if (endstop(Y_END)) { halt("ENDSTOP_Y"); evt("EVT:ENDSTOP_Y"); return; }
        if (endstop(Z_END)) { halt("ENDSTOP_Z"); evt("EVT:ENDSTOP_Z"); return; }
    }
}

// ===================== HOMING (3-phase per axis: fast→backoff→slow) =====================
void startHome() {
    setMag(false); homed = false; enableMotors(true); setRelay(true);
    state = S_HOME_Z1;
    sZ.setMaxSpeed(HOME_FAST * STEPS_MM_Z);
    sZ.moveTo(-999999);
    evt("HOMING", "Z_FAST");
}

void handleHome() {
    switch (state) {
        case S_HOME_Z1:
            if (endstop(Z_END)) {
                sZ.stop(); sZ.setCurrentPosition(0);
                sZ.moveTo((long)(HOME_BACKOFF * STEPS_MM_Z));
                state = S_HOME_Z_BACK; evt("HOMING", "Z_BACKOFF");
            } break;
        case S_HOME_Z_BACK:
            if (!sZ.isRunning()) {
                sZ.setMaxSpeed(HOME_SLOW * STEPS_MM_Z);
                sZ.moveTo(-999999);
                state = S_HOME_Z2; evt("HOMING", "Z_SLOW");
            } break;
        case S_HOME_Z2:
            if (endstop(Z_END)) {
                sZ.stop(); sZ.setCurrentPosition(0);
                sZ.setMaxSpeed(Z_SPEED * STEPS_MM_Z);
                // Start X
                sX.setMaxSpeed(HOME_FAST * STEPS_MM_XY);
                sX.moveTo(-999999);
                state = S_HOME_X1; evt("HOMING", "X_FAST");
            } break;
        case S_HOME_X1:
            if (endstop(X_END)) {
                sX.stop(); sX.setCurrentPosition(0);
                sX.moveTo((long)(HOME_BACKOFF * STEPS_MM_XY));
                state = S_HOME_X_BACK; evt("HOMING", "X_BACKOFF");
            } break;
        case S_HOME_X_BACK:
            if (!sX.isRunning()) {
                sX.setMaxSpeed(HOME_SLOW * STEPS_MM_XY);
                sX.moveTo(-999999);
                state = S_HOME_X2; evt("HOMING", "X_SLOW");
            } break;
        case S_HOME_X2:
            if (endstop(X_END)) {
                sX.stop(); sX.setCurrentPosition(0);
                sX.setMaxSpeed(XY_SPEED * STEPS_MM_XY);
                sY.setMaxSpeed(HOME_FAST * STEPS_MM_XY);
                sY.moveTo(-999999);
                state = S_HOME_Y1; evt("HOMING", "Y_FAST");
            } break;
        case S_HOME_Y1:
            if (endstop(Y_END)) {
                sY.stop(); sY.setCurrentPosition(0);
                sY.moveTo((long)(HOME_BACKOFF * STEPS_MM_XY));
                state = S_HOME_Y_BACK; evt("HOMING", "Y_BACKOFF");
            } break;
        case S_HOME_Y_BACK:
            if (!sY.isRunning()) {
                sY.setMaxSpeed(HOME_SLOW * STEPS_MM_XY);
                sY.moveTo(-999999);
                state = S_HOME_Y2; evt("HOMING", "Y_SLOW");
            } break;
        case S_HOME_Y2:
            if (endstop(Y_END)) {
                sY.stop(); sY.setCurrentPosition(0);
                sY.setMaxSpeed(XY_SPEED * STEPS_MM_XY);
                state = S_HOME_FINAL;
            } break;
        case S_HOME_FINAL:
            if (!sX.isRunning() && !sY.isRunning() && !sZ.isRunning()) {
                homed = true; state = S_IDLE; evt("HOMED", "ALL");
            } break;
        default: break;
    }
}

// ===================== MOVE HANDLING =====================
void handleMove() {
    if (!sX.isRunning() && !sY.isRunning() && !sZ.isRunning()) {
        state = S_IDLE;
        char d[64];
        snprintf(d, sizeof(d), "AT X%.2f Y%.2f Z%.2f",
            sX.currentPosition() / STEPS_MM_XY,
            sY.currentPosition() / STEPS_MM_XY,
            sZ.currentPosition() / STEPS_MM_Z);
        ack(moveSeq, "DONE", d);
    }
}

// ===================== COMMAND PROCESSING =====================
void processCmd(const char* cmd) {
    uint16_t seq = 0;
    char ct[16] = {0};
    if (sscanf(cmd, "%hu %15s", &seq, ct) < 2) { ack(0, "ERR:PARSE"); return; }

    // Always-available commands
    if (!strcmp(ct, "PING"))   { ack(seq, "OK", "PONG"); return; }
    if (!strcmp(ct, "HALT"))   { halt("CMD"); ack(seq, "OK", "HALTED"); return; }
    if (!strcmp(ct, "STATUS")) {
        char d[96];
        const char* sn[] = {"BOOT","NEED_HOME","HOME_Z1","HOME_ZB","HOME_Z2",
            "HOME_X1","HOME_XB","HOME_X2","HOME_Y1","HOME_YB","HOME_Y2",
            "HOME_F","IDLE","MOVING","HALT","ERROR"};
        snprintf(d, sizeof(d), "x=%.2f y=%.2f z=%.2f state=%s mag=%d homed=%d",
            sX.currentPosition() / STEPS_MM_XY,
            sY.currentPosition() / STEPS_MM_XY,
            sZ.currentPosition() / STEPS_MM_Z,
            sn[state], magOn ? 1 : 0, homed ? 1 : 0);
        ack(seq, "OK", d); return;
    }
    if (!strcmp(ct, "HOME")) {
        if (state == S_MOVING) { ack(seq, "ERR:BUSY"); return; }
        startHome(); ack(seq, "OK", "HOMING"); return;
    }
    if (!strcmp(ct, "RELAY")) {
        const char* p = cmd; skipTo(&p, 2);
        if (!strncmp(p, "ON", 2)) setRelay(true);
        else if (!strncmp(p, "OFF", 3)) setRelay(false);
        ack(seq, "OK", relayOn ? "RELAY_ON" : "RELAY_OFF"); return;
    }

    // Homed-only commands
    if (!homed) { ack(seq, "ERR:NOT_HOMED"); return; }
    if (state == S_HALT || state == S_ERROR) { ack(seq, "ERR:HALTED"); return; }

    if (!strcmp(ct, "MOVE")) {
        if (state == S_MOVING) { ack(seq, "ERR:BUSY"); return; }
        float mx = -1, my = -1, mz = -1;
        const char* p = cmd; skipTo(&p, 2);
        parseXYZ(p, &mx, &my, &mz);
        if (mx < 0 || my < 0 || mz < 0) { ack(seq, "ERR:PARSE"); return; }
        mx = constrain(mx, 0, X_MAX_MM);
        my = constrain(my, 0, Y_MAX_MM);
        mz = constrain(mz, 0, Z_MAX_MM);
        startMove(mx, my, mz, seq);
        return;
    }

    if (!strcmp(ct, "PICK")) {
        // PICK <sq> — move to square, lower Z, magnet on, lift
        // Simplified: host handles sequencing; PICK just enables magnet
        if (state == S_MOVING) { ack(seq, "ERR:BUSY"); return; }
        setMag(true);
        ack(seq, "OK", "MAG_ON"); return;
    }

    if (!strcmp(ct, "PLACE")) {
        if (state == S_MOVING) { ack(seq, "ERR:BUSY"); return; }
        setMag(false);
        ack(seq, "OK", "MAG_OFF"); return;
    }

    if (!strcmp(ct, "MAG")) {
        const char* p = cmd; skipTo(&p, 2);
        if (!strncmp(p, "ON", 2)) setMag(true);
        else if (!strncmp(p, "OFF", 3)) setMag(false);
        ack(seq, "OK", magOn ? "MAG_ON" : "MAG_OFF"); return;
    }

    if (!strcmp(ct, "JOG")) {
        if (state == S_MOVING) { ack(seq, "ERR:BUSY"); return; }
        char axis = 0; long steps = 0;
        const char* p = cmd; skipTo(&p, 2);
        sscanf(p, "%c %ld", &axis, &steps);
        AccelStepper* s = nullptr;
        if (axis == 'X' || axis == 'x') s = &sX;
        else if (axis == 'Y' || axis == 'y') s = &sY;
        else if (axis == 'Z' || axis == 'z') s = &sZ;
        if (!s) { ack(seq, "ERR:AXIS"); return; }
        s->move(steps);
        moveSeq = seq; state = S_MOVING;
        moveStartMs = millis();
        moveTimeoutMs = (unsigned long)(abs(steps) / (XY_SPEED * STEPS_MM_XY) * 1000 * MOTION_WDT_MULT) + 2000;
        ack(seq, "OK", "JOGGING"); return;
    }

    if (!strcmp(ct, "SETSPEED")) {
        char axis = 0; float spd = 0;
        const char* p = cmd; skipTo(&p, 2);
        sscanf(p, "%c %f", &axis, &spd);
        if (axis == 'X' || axis == 'x') sX.setMaxSpeed(spd);
        else if (axis == 'Y' || axis == 'y') sY.setMaxSpeed(spd);
        else if (axis == 'Z' || axis == 'z') sZ.setMaxSpeed(spd);
        else { ack(seq, "ERR:AXIS"); return; }
        ack(seq, "OK", "SPEED_SET"); return;
    }

    ack(seq, "ERR:UNKNOWN");
}

// ===================== HELPERS =====================
void startMove(float mx, float my, float mz, uint16_t seq) {
    sX.moveTo((long)(mx * STEPS_MM_XY));
    sY.moveTo((long)(my * STEPS_MM_XY));
    sZ.moveTo((long)(mz * STEPS_MM_Z));
    moveSeq = seq; state = S_MOVING;
    moveStartMs = millis();
    // Estimate move time from max axis distance
    float dx = abs(mx - sX.currentPosition() / STEPS_MM_XY);
    float dy = abs(my - sY.currentPosition() / STEPS_MM_XY);
    float dz = abs(mz - sZ.currentPosition() / STEPS_MM_Z);
    float maxDist = max(max(dx, dy), dz);
    moveTimeoutMs = (unsigned long)(maxDist / min(XY_SPEED, Z_SPEED) * 1000 * MOTION_WDT_MULT) + 3000;
    char d[64];
    snprintf(d, sizeof(d), "TO X%.1f Y%.1f Z%.1f", mx, my, mz);
    ack(seq, "OK", d);
}

void skipTo(const char** p, int words) {
    for (int i = 0; i < words; i++) {
        while (**p && **p != ' ') (*p)++;
        while (**p == ' ') (*p)++;
    }
}

void parseXYZ(const char* p, float* x, float* y, float* z) {
    while (*p) {
        if (*p == 'X' || *p == 'x') *x = atof(p + 1);
        else if (*p == 'Y' || *p == 'y') *y = atof(p + 1);
        else if (*p == 'Z' || *p == 'z') *z = atof(p + 1);
        while (*p && *p != ' ') p++;
        while (*p == ' ') p++;
    }
}

void halt(const char* r) {
    sX.stop(); sY.stop(); sZ.stop();
    setMag(false);
    state = S_HALT;
    evt("HALT", r);
}

void setMag(bool on) {
    magOn = on;
    digitalWrite(MAG_PIN, on ? HIGH : LOW);
}

void setRelay(bool on) {
    relayOn = on;
    digitalWrite(RELAY_PIN, on ? HIGH : LOW);
    if (!on) { sX.stop(); sY.stop(); sZ.stop(); }
}

bool endstop(int pin) { return digitalRead(pin) == LOW; }

void enableMotors(bool en) {
    // A4988: ENABLE is active LOW
    digitalWrite(X_EN, en ? LOW : HIGH);
    digitalWrite(Y_EN, en ? LOW : HIGH);
    digitalWrite(Z_EN, en ? LOW : HIGH);
}

void ack(uint16_t s, const char* st, const char* d) {
    Serial.print(st); Serial.print(" "); Serial.print(s);
    if (d && d[0]) { Serial.print(" "); Serial.print(d); }
    Serial.println();
}

void evt(const char* e, const char* d) {
    Serial.print("EVT:"); Serial.print(e);
    if (d && d[0]) { Serial.print(" "); Serial.print(d); }
    Serial.println();
}
