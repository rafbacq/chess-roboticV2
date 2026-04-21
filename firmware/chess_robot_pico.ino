/*
 * Chess Gantry Robot — Raspberry Pi Pico Firmware
 * ================================================
 *
 * Stepper-based XYZ gantry with electromagnet end-effector.
 *
 * Hardware:
 *   - 3× NEMA17 steppers via A4988/DRV8825 drivers (STEP/DIR)
 *   - 3× Mechanical endstops (NC, active-low with internal pullup)
 *   - 1× Electromagnet via IRLZ44N MOSFET + 1N4007 flyback diode
 *   - 1× Spare driver channel (disabled, for future captured-piece bin)
 *   - Serial over USB (115200 baud, structured protocol)
 *
 * Pin Map (RP2040, no UART0 conflict on GP0/GP1):
 *   X Stepper: STEP=GP2, DIR=GP3
 *   Y Stepper: STEP=GP4, DIR=GP5
 *   Z Stepper: STEP=GP6, DIR=GP7
 *   Spare:     STEP=GP8, DIR=GP9 (disabled)
 *
 *   X Endstop: GP10 (NC, pulled high — LOW when triggered)
 *   Y Endstop: GP11
 *   Z Endstop: GP12
 *
 *   Electromagnet: GP13 (MOSFET gate, active-high)
 *   LED Status:    GP25 (onboard LED)
 *
 *   Microstepping select (optional, directly wired to MS1/MS2/MS3):
 *     A4988: MS1=GP14, MS2=GP15, MS3=GP16 (for X axis; Y/Z wired similarly or hardwired)
 *     Default: 1/16 microstepping (MS1=HIGH, MS2=HIGH, MS3=LOW for A4988)
 *
 * Safety:
 *   - Watchdog: 2-second timeout, auto-reset on command receipt
 *   - Endstop polling at every step (hard stop on trigger during motion)
 *   - Electromagnet auto-off on HALT or watchdog reset
 *   - Homing required before any motion command accepted
 *   - Per-axis max travel limits enforced in software
 *
 * Protocol: See PROTOCOL.md
 *
 * Dependencies:
 *   - AccelStepper library (via PlatformIO / Arduino Library Manager)
 *   - Arduino-Pico core (Earle Philhower)
 */

#include <AccelStepper.h>

// ============================================================
// Pin Definitions — CRITICAL: GP0/GP1 reserved for USB serial
// ============================================================
#define X_STEP_PIN    2
#define X_DIR_PIN     3
#define Y_STEP_PIN    4
#define Y_DIR_PIN     5
#define Z_STEP_PIN    6
#define Z_DIR_PIN     7
#define SPARE_STEP    8   // Disabled, future use
#define SPARE_DIR     9

#define X_ENDSTOP_PIN 10  // NC endstop, active LOW
#define Y_ENDSTOP_PIN 11
#define Z_ENDSTOP_PIN 12

#define MAGNET_PIN    13  // IRLZ44N gate — HIGH = energized
#define LED_PIN       25  // Onboard LED

// Microstepping select (optional, can be hardwired)
#define MS1_PIN       14
#define MS2_PIN       15
#define MS3_PIN       16

// ============================================================
// Motion Parameters
// ============================================================
// NEMA17 1.8° = 200 full steps/rev
// With 1/16 microstepping = 3200 microsteps/rev
// GT2 belt 20T pulley = 40mm/rev
// → 3200 / 40mm = 80 microsteps/mm
#define STEPS_PER_MM_XY     80.0f
#define STEPS_PER_MM_Z      80.0f   // Adjust for Z lead screw if different

// Travel limits (mm) — measured from home (0,0,0)
#define X_MAX_MM            300.0f  // ~12" board + margins
#define Y_MAX_MM            300.0f
#define Z_MAX_MM             60.0f  // Z only needs ~40mm travel

// Speed / acceleration
#define XY_MAX_SPEED_MMPS   100.0f  // mm/sec
#define Z_MAX_SPEED_MMPS     30.0f
#define XY_ACCEL_MMPS2      200.0f  // mm/sec²
#define Z_ACCEL_MMPS2       100.0f

// Homing
#define HOMING_SPEED_MMPS    20.0f  // Slow approach
#define HOMING_BACKOFF_MM     3.0f  // Back off after trigger

// Watchdog
#define WATCHDOG_TIMEOUT_MS  2000   // 2 seconds

// ============================================================
// State Machine
// ============================================================
enum SystemState {
    STATE_BOOT,          // Initial startup
    STATE_NEED_HOME,     // Homing required before motion
    STATE_HOMING_X,      // Homing X axis
    STATE_HOMING_Y,
    STATE_HOMING_Z,
    STATE_HOMING_BACKOFF,
    STATE_IDLE,          // Ready for commands
    STATE_MOVING,        // Executing a move
    STATE_HALT,          // Emergency stop
    STATE_ERROR          // Unrecoverable error
};

// ============================================================
// Globals
// ============================================================
AccelStepper stepperX(AccelStepper::DRIVER, X_STEP_PIN, X_DIR_PIN);
AccelStepper stepperY(AccelStepper::DRIVER, Y_STEP_PIN, Y_DIR_PIN);
AccelStepper stepperZ(AccelStepper::DRIVER, Z_STEP_PIN, Z_DIR_PIN);

SystemState state = STATE_BOOT;
SystemState homingNextAxis = STATE_IDLE;

unsigned long lastCommandMs = 0;
unsigned long lastHeartbeatMs = 0;
bool magnetOn = false;
bool homed = false;

// Current position tracking (in mm)
float posX_mm = 0.0f;
float posY_mm = 0.0f;
float posZ_mm = 0.0f;

// Command parsing
#define CMD_BUF_SIZE 128
char cmdBuffer[CMD_BUF_SIZE];
int cmdIndex = 0;
uint16_t cmdSeq = 0;  // Command sequence number for ack

// Move target (for non-blocking move tracking)
long targetStepsX = 0;
long targetStepsY = 0;
long targetStepsZ = 0;
uint16_t moveSeq = 0;  // Sequence of current move

// ============================================================
// Forward declarations
// ============================================================
void processCommand(const char* cmd);
void sendAck(uint16_t seq, const char* status, const char* detail = "");
void sendEvent(const char* event, const char* detail = "");
void haltAll(const char* reason);
void setMagnet(bool on);
bool isEndstopTriggered(int pin);
void configMicrostepping();
void startHoming();
void handleHoming();
void handleMoving();

// ============================================================
// Setup
// ============================================================
void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {
        // Wait up to 3s for USB serial
    }

    // LED
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH);

    // Endstops — INPUT_PULLUP for NC switches
    pinMode(X_ENDSTOP_PIN, INPUT_PULLUP);
    pinMode(Y_ENDSTOP_PIN, INPUT_PULLUP);
    pinMode(Z_ENDSTOP_PIN, INPUT_PULLUP);

    // Magnet — start OFF
    pinMode(MAGNET_PIN, OUTPUT);
    digitalWrite(MAGNET_PIN, LOW);

    // Microstepping select
    configMicrostepping();

    // Configure steppers (values in steps, not mm)
    stepperX.setMaxSpeed(XY_MAX_SPEED_MMPS * STEPS_PER_MM_XY);
    stepperX.setAcceleration(XY_ACCEL_MMPS2 * STEPS_PER_MM_XY);

    stepperY.setMaxSpeed(XY_MAX_SPEED_MMPS * STEPS_PER_MM_XY);
    stepperY.setAcceleration(XY_ACCEL_MMPS2 * STEPS_PER_MM_XY);

    stepperZ.setMaxSpeed(Z_MAX_SPEED_MMPS * STEPS_PER_MM_Z);
    stepperZ.setAcceleration(Z_ACCEL_MMPS2 * STEPS_PER_MM_Z);

    state = STATE_NEED_HOME;
    lastCommandMs = millis();

    sendEvent("BOOT", "v2.0 stepper gantry ready");
}

// ============================================================
// Main Loop — fully non-blocking
// ============================================================
void loop() {
    unsigned long now = millis();

    // --- Read serial commands (non-blocking) ---
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (cmdIndex > 0) {
                cmdBuffer[cmdIndex] = '\0';
                processCommand(cmdBuffer);
                cmdIndex = 0;
                lastCommandMs = now;  // Reset watchdog
            }
        } else if (cmdIndex < CMD_BUF_SIZE - 1) {
            cmdBuffer[cmdIndex++] = c;
        }
    }

    // --- Watchdog ---
    if (state == STATE_IDLE || state == STATE_MOVING) {
        if ((now - lastCommandMs) > WATCHDOG_TIMEOUT_MS) {
            haltAll("WATCHDOG_TIMEOUT");
        }
    }

    // --- State machine tick ---
    switch (state) {
        case STATE_BOOT:
        case STATE_NEED_HOME:
            // Blink LED slowly to indicate need-home
            digitalWrite(LED_PIN, (now / 500) % 2);
            break;

        case STATE_HOMING_X:
        case STATE_HOMING_Y:
        case STATE_HOMING_Z:
        case STATE_HOMING_BACKOFF:
            handleHoming();
            break;

        case STATE_IDLE:
            // Solid LED
            digitalWrite(LED_PIN, HIGH);
            break;

        case STATE_MOVING:
            handleMoving();
            break;

        case STATE_HALT:
            // Fast blink
            digitalWrite(LED_PIN, (now / 100) % 2);
            setMagnet(false);  // Safety: magnet off on halt
            break;

        case STATE_ERROR:
            // Very fast blink
            digitalWrite(LED_PIN, (now / 50) % 2);
            setMagnet(false);
            break;
    }

    // --- Run steppers (non-blocking) ---
    stepperX.run();
    stepperY.run();
    stepperZ.run();

    // --- Heartbeat every 500ms ---
    if ((now - lastHeartbeatMs) > 500) {
        lastHeartbeatMs = now;
        // Only send heartbeat in IDLE or MOVING
        if (state == STATE_IDLE || state == STATE_MOVING) {
            char buf[80];
            snprintf(buf, sizeof(buf), "X%.2f Y%.2f Z%.2f M%d S%d",
                     stepperX.currentPosition() / STEPS_PER_MM_XY,
                     stepperY.currentPosition() / STEPS_PER_MM_XY,
                     stepperZ.currentPosition() / STEPS_PER_MM_Z,
                     magnetOn ? 1 : 0,
                     (int)state);
            sendEvent("POS", buf);
        }
    }
}

// ============================================================
// Command Processing
// ============================================================
void processCommand(const char* cmd) {
    // Protocol: SEQ CMD [ARGS...]
    // Example: "42 MOVE X100.0 Y200.0 Z10.0"
    // Example: "43 HOME"
    // Example: "44 MAGNET 1"
    // Example: "45 HALT"
    // Example: "46 PING"
    // Example: "47 STATUS"
    // Example: "48 RESET"

    uint16_t seq = 0;
    char cmdType[16] = {0};
    int parsed = sscanf(cmd, "%hu %15s", &seq, cmdType);

    if (parsed < 2) {
        sendAck(0, "ERR", "PARSE_FAIL");
        return;
    }

    // --- PING: always respond ---
    if (strcmp(cmdType, "PING") == 0) {
        sendAck(seq, "OK", "PONG");
        return;
    }

    // --- STATUS: report current state ---
    if (strcmp(cmdType, "STATUS") == 0) {
        char detail[80];
        snprintf(detail, sizeof(detail), "STATE=%d HOMED=%d X=%.2f Y=%.2f Z=%.2f MAG=%d",
                 (int)state, homed ? 1 : 0,
                 stepperX.currentPosition() / STEPS_PER_MM_XY,
                 stepperY.currentPosition() / STEPS_PER_MM_XY,
                 stepperZ.currentPosition() / STEPS_PER_MM_Z,
                 magnetOn ? 1 : 0);
        sendAck(seq, "OK", detail);
        return;
    }

    // --- HALT: emergency stop from any state ---
    if (strcmp(cmdType, "HALT") == 0) {
        haltAll("CMD_HALT");
        sendAck(seq, "OK", "HALTED");
        return;
    }

    // --- RESET: recover from HALT/ERROR, require re-homing ---
    if (strcmp(cmdType, "RESET") == 0) {
        setMagnet(false);
        stepperX.stop();
        stepperY.stop();
        stepperZ.stop();
        homed = false;
        state = STATE_NEED_HOME;
        sendAck(seq, "OK", "RESET_NEED_HOME");
        return;
    }

    // --- HOME: begin homing sequence ---
    if (strcmp(cmdType, "HOME") == 0) {
        if (state == STATE_MOVING) {
            sendAck(seq, "ERR", "BUSY_MOVING");
            return;
        }
        startHoming();
        sendAck(seq, "OK", "HOMING_STARTED");
        return;
    }

    // --- Commands below require homed state ---
    if (!homed) {
        sendAck(seq, "ERR", "NOT_HOMED");
        return;
    }

    if (state == STATE_HALT || state == STATE_ERROR) {
        sendAck(seq, "ERR", "IN_HALT_OR_ERROR");
        return;
    }

    // --- MOVE X<mm> Y<mm> Z<mm>: absolute move ---
    if (strcmp(cmdType, "MOVE") == 0) {
        if (state == STATE_MOVING) {
            sendAck(seq, "ERR", "BUSY_MOVING");
            return;
        }

        float mx = -1, my = -1, mz = -1;
        const char* p = cmd;
        // Skip seq and MOVE
        while (*p && *p != ' ') p++; // skip seq
        while (*p == ' ') p++;
        while (*p && *p != ' ') p++; // skip MOVE
        while (*p == ' ') p++;

        // Parse X, Y, Z arguments
        while (*p) {
            if (*p == 'X' || *p == 'x') mx = atof(p + 1);
            else if (*p == 'Y' || *p == 'y') my = atof(p + 1);
            else if (*p == 'Z' || *p == 'z') mz = atof(p + 1);
            // Advance to next space
            while (*p && *p != ' ') p++;
            while (*p == ' ') p++;
        }

        if (mx < 0 || my < 0 || mz < 0) {
            sendAck(seq, "ERR", "MOVE_PARSE_FAIL");
            return;
        }

        // Clamp to travel limits
        mx = constrain(mx, 0.0f, X_MAX_MM);
        my = constrain(my, 0.0f, Y_MAX_MM);
        mz = constrain(mz, 0.0f, Z_MAX_MM);

        targetStepsX = (long)(mx * STEPS_PER_MM_XY);
        targetStepsY = (long)(my * STEPS_PER_MM_XY);
        targetStepsZ = (long)(mz * STEPS_PER_MM_Z);

        stepperX.moveTo(targetStepsX);
        stepperY.moveTo(targetStepsY);
        stepperZ.moveTo(targetStepsZ);

        moveSeq = seq;
        state = STATE_MOVING;

        char detail[64];
        snprintf(detail, sizeof(detail), "TARGET X%.1f Y%.1f Z%.1f", mx, my, mz);
        sendAck(seq, "OK", detail);
        return;
    }

    // --- MAGNET 0|1: electromagnet control ---
    if (strcmp(cmdType, "MAGNET") == 0) {
        int val = 0;
        const char* p = cmd;
        // Skip to value
        while (*p && *p != ' ') p++; while (*p == ' ') p++;
        while (*p && *p != ' ') p++; while (*p == ' ') p++;
        val = atoi(p);

        setMagnet(val != 0);
        sendAck(seq, "OK", val ? "MAGNET_ON" : "MAGNET_OFF");
        return;
    }

    // --- Unknown command ---
    sendAck(seq, "ERR", "UNKNOWN_CMD");
}

// ============================================================
// Homing
// ============================================================
void startHoming() {
    setMagnet(false);
    homed = false;

    // Home Z first (safety — lift head), then X, then Y
    state = STATE_HOMING_Z;
    stepperZ.setMaxSpeed(HOMING_SPEED_MMPS * STEPS_PER_MM_Z);
    stepperZ.moveTo(-100000);  // Move toward endstop (negative = toward home)

    sendEvent("HOMING", "Z_START");
}

void handleHoming() {
    int endstopPin;
    AccelStepper* stepper;

    switch (state) {
        case STATE_HOMING_Z:
            endstopPin = Z_ENDSTOP_PIN;
            stepper = &stepperZ;
            break;
        case STATE_HOMING_X:
            endstopPin = X_ENDSTOP_PIN;
            stepper = &stepperX;
            break;
        case STATE_HOMING_Y:
            endstopPin = Y_ENDSTOP_PIN;
            stepper = &stepperY;
            break;
        case STATE_HOMING_BACKOFF:
            // Back off from endstop
            if (!stepperX.isRunning() && !stepperY.isRunning() && !stepperZ.isRunning()) {
                // All backoffs complete — set positions to 0
                stepperX.setCurrentPosition(0);
                stepperY.setCurrentPosition(0);
                stepperZ.setCurrentPosition(0);

                // Restore normal speeds
                stepperX.setMaxSpeed(XY_MAX_SPEED_MMPS * STEPS_PER_MM_XY);
                stepperY.setMaxSpeed(XY_MAX_SPEED_MMPS * STEPS_PER_MM_XY);
                stepperZ.setMaxSpeed(Z_MAX_SPEED_MMPS * STEPS_PER_MM_Z);

                homed = true;
                state = STATE_IDLE;
                sendEvent("HOMED", "ALL_AXES");
            }
            return;
        default:
            return;
    }

    // Check if endstop triggered
    if (isEndstopTriggered(endstopPin)) {
        stepper->stop();
        stepper->setCurrentPosition(0);

        // Backoff
        long backoffSteps = (long)(HOMING_BACKOFF_MM *
            ((state == STATE_HOMING_Z) ? STEPS_PER_MM_Z : STEPS_PER_MM_XY));
        stepper->moveTo(backoffSteps);

        // Transition to next axis or backoff
        if (state == STATE_HOMING_Z) {
            sendEvent("HOMING", "Z_HIT");
            state = STATE_HOMING_X;
            stepperX.setMaxSpeed(HOMING_SPEED_MMPS * STEPS_PER_MM_XY);
            stepperX.moveTo(-100000);
            sendEvent("HOMING", "X_START");
        } else if (state == STATE_HOMING_X) {
            sendEvent("HOMING", "X_HIT");
            state = STATE_HOMING_Y;
            stepperY.setMaxSpeed(HOMING_SPEED_MMPS * STEPS_PER_MM_XY);
            stepperY.moveTo(-100000);
            sendEvent("HOMING", "Y_START");
        } else if (state == STATE_HOMING_Y) {
            sendEvent("HOMING", "Y_HIT");
            // Start backoff for all axes
            long backXY = (long)(HOMING_BACKOFF_MM * STEPS_PER_MM_XY);
            long backZ  = (long)(HOMING_BACKOFF_MM * STEPS_PER_MM_Z);
            stepperX.moveTo(backXY);
            stepperY.moveTo(backXY);
            stepperZ.moveTo(backZ);
            state = STATE_HOMING_BACKOFF;
            sendEvent("HOMING", "BACKOFF_START");
        }
    }
}

// ============================================================
// Movement (non-blocking)
// ============================================================
void handleMoving() {
    // Check endstops during motion (safety)
    if (stepperX.isRunning() && isEndstopTriggered(X_ENDSTOP_PIN)) {
        haltAll("X_ENDSTOP_HIT_DURING_MOVE");
        return;
    }
    if (stepperY.isRunning() && isEndstopTriggered(Y_ENDSTOP_PIN)) {
        haltAll("Y_ENDSTOP_HIT_DURING_MOVE");
        return;
    }
    if (stepperZ.isRunning() && isEndstopTriggered(Z_ENDSTOP_PIN)) {
        haltAll("Z_ENDSTOP_HIT_DURING_MOVE");
        return;
    }

    // Check if all axes have reached target
    if (!stepperX.isRunning() && !stepperY.isRunning() && !stepperZ.isRunning()) {
        state = STATE_IDLE;
        char detail[64];
        snprintf(detail, sizeof(detail), "AT X%.2f Y%.2f Z%.2f",
                 stepperX.currentPosition() / STEPS_PER_MM_XY,
                 stepperY.currentPosition() / STEPS_PER_MM_XY,
                 stepperZ.currentPosition() / STEPS_PER_MM_Z);
        sendAck(moveSeq, "DONE", detail);
        sendEvent("MOVE_DONE", detail);
    }
}

// ============================================================
// Safety
// ============================================================
void haltAll(const char* reason) {
    // Immediate stop — decelerate to zero
    stepperX.stop();
    stepperY.stop();
    stepperZ.stop();

    // CRITICAL: magnet OFF on halt to prevent piece from sticking
    // mid-air and falling when power is lost
    setMagnet(false);

    state = STATE_HALT;
    sendEvent("HALT", reason);
}

void setMagnet(bool on) {
    magnetOn = on;
    digitalWrite(MAGNET_PIN, on ? HIGH : LOW);
    // NOTE: IRLZ44N MOSFET with 1N4007 flyback diode across the coil.
    // The flyback diode (cathode to +V, anode to drain) prevents
    // back-EMF from damaging the MOSFET when the coil de-energizes.
}

bool isEndstopTriggered(int pin) {
    // NC endstop: normally HIGH (closed circuit through pullup).
    // When triggered (switch opens), the pin reads LOW.
    // We invert so triggered = true.
    return digitalRead(pin) == LOW;
}

// ============================================================
// Microstepping Configuration
// ============================================================
void configMicrostepping() {
    // A4988 1/16 microstepping: MS1=HIGH, MS2=HIGH, MS3=LOW
    // DRV8825 1/16 microstepping: M0=LOW, M1=LOW, M2=HIGH
    // Default to A4988 config; change if using DRV8825
    #ifdef USE_DRV8825
        pinMode(MS1_PIN, OUTPUT); digitalWrite(MS1_PIN, LOW);
        pinMode(MS2_PIN, OUTPUT); digitalWrite(MS2_PIN, LOW);
        pinMode(MS3_PIN, OUTPUT); digitalWrite(MS3_PIN, HIGH);
    #else
        // A4988 default
        pinMode(MS1_PIN, OUTPUT); digitalWrite(MS1_PIN, HIGH);
        pinMode(MS2_PIN, OUTPUT); digitalWrite(MS2_PIN, HIGH);
        pinMode(MS3_PIN, OUTPUT); digitalWrite(MS3_PIN, LOW);
    #endif
}

// ============================================================
// Protocol Output
// ============================================================
void sendAck(uint16_t seq, const char* status, const char* detail) {
    // Format: "ACK <seq> <status> [detail]\n"
    Serial.print("ACK ");
    Serial.print(seq);
    Serial.print(" ");
    Serial.print(status);
    if (detail && detail[0]) {
        Serial.print(" ");
        Serial.print(detail);
    }
    Serial.println();
}

void sendEvent(const char* event, const char* detail) {
    // Format: "EVT <event> [detail]\n"
    Serial.print("EVT ");
    Serial.print(event);
    if (detail && detail[0]) {
        Serial.print(" ");
        Serial.print(detail);
    }
    Serial.println();
}
