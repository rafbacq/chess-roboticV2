# Serial Protocol — Chess Gantry Robot v2.0

## Overview

Line-based ASCII protocol over USB serial at **115200 baud**.
Each line ends with `\n`. Commands are sequence-numbered for reliable acking.

## Message Types

| Direction | Prefix | Format |
|-----------|--------|--------|
| Host → Pico | (none) | `<seq> <CMD> [args...]` |
| Pico → Host | `ACK` | `ACK <seq> <OK\|ERR\|DONE> [detail]` |
| Pico → Host | `EVT` | `EVT <event> [detail]` |

## Commands (Host → Pico)

### PING
```
42 PING
→ ACK 42 OK PONG
```
Keepalive / connectivity check.

### STATUS
```
43 STATUS
→ ACK 43 OK STATE=4 HOMED=1 X=100.00 Y=150.00 Z=10.00 MAG=0
```
Returns current state, position, and magnet status.

### HOME
```
44 HOME
→ ACK 44 OK HOMING_STARTED
→ EVT HOMING Z_START
→ EVT HOMING Z_HIT
→ EVT HOMING X_START
→ EVT HOMING X_HIT
→ EVT HOMING Y_START
→ EVT HOMING Y_HIT
→ EVT HOMING BACKOFF_START
→ EVT HOMED ALL_AXES
```
Home all axes (Z first for safety, then X, then Y).
Homing uses endstop detection with 3mm backoff.

### MOVE
```
45 MOVE X100.0 Y200.0 Z10.0
→ ACK 45 OK TARGET X100.0 Y200.0 Z10.0
→ EVT MOVE_DONE AT X100.00 Y200.00 Z10.00
```
Absolute move to (X, Y, Z) in millimeters.
Coordinates are clamped to travel limits.
Motion uses acceleration ramping (AccelStepper).

**Requires**: Homing completed.
**Rejects**: If already moving (ACK ERR BUSY_MOVING).

### MAGNET
```
46 MAGNET 1
→ ACK 46 OK MAGNET_ON
47 MAGNET 0
→ ACK 47 OK MAGNET_OFF
```
Energize (1) or de-energize (0) the electromagnet.

**Safety**: Magnet is automatically turned OFF on HALT or watchdog timeout.

### HALT
```
48 HALT
→ ACK 48 OK HALTED
→ EVT HALT CMD_HALT
```
Emergency stop. Decelerates all axes, turns magnet OFF.
Requires RESET + HOME to resume.

### RESET
```
49 RESET
→ ACK 49 OK RESET_NEED_HOME
```
Recover from HALT or ERROR state. Requires re-homing before motion.

## Events (Pico → Host)

| Event | Detail | Meaning |
|-------|--------|---------|
| `BOOT` | `v2.0 stepper gantry ready` | Firmware started |
| `POS` | `X<mm> Y<mm> Z<mm> M<0\|1> S<state>` | Position heartbeat (every 500ms) |
| `HOMING` | `<axis>_START\|<axis>_HIT\|BACKOFF_START` | Homing progress |
| `HOMED` | `ALL_AXES` | Homing complete |
| `MOVE_DONE` | `AT X<mm> Y<mm> Z<mm>` | Move completed |
| `HALT` | `<reason>` | Emergency stop triggered |

## State Machine

```
BOOT → NEED_HOME → HOMING_Z → HOMING_X → HOMING_Y → HOMING_BACKOFF → IDLE
IDLE → MOVING → IDLE
(any) → HALT → (RESET) → NEED_HOME
(any) → ERROR
```

## Error Codes

| Error | Meaning |
|-------|---------|
| `PARSE_FAIL` | Command format invalid |
| `NOT_HOMED` | Motion command before homing |
| `BUSY_MOVING` | Move command while already moving |
| `IN_HALT_OR_ERROR` | Command rejected in HALT/ERROR |
| `MOVE_PARSE_FAIL` | Could not parse X/Y/Z from MOVE |
| `UNKNOWN_CMD` | Unrecognized command |

## Watchdog

The firmware has a **2-second watchdog** timer. If no command is received
within 2 seconds during IDLE or MOVING states, the firmware triggers
an automatic HALT.

**The host must send PING or STATUS at least every 1.5 seconds as a keepalive.**

## Timing

| Operation | Typical Duration |
|-----------|-----------------|
| Homing (all axes) | 15–30 seconds |
| XY move (150mm) | ~2 seconds |
| Z move (40mm) | ~1.5 seconds |
| Magnet toggle | <1ms |
| HALT response | <100ms (deceleration) |

## Hardware Notes

- **Endstops**: NC (normally closed), active-low with internal pullup.
  Triggered (switch opens) = pin reads LOW.
- **Electromagnet**: IRLZ44N MOSFET (logic-level gate).
  1N4007 flyback diode across coil (cathode to +V, anode to MOSFET drain).
- **Microstepping**: Default 1/16 (A4988: MS1=HIGH, MS2=HIGH, MS3=LOW).
- **Steps/mm**: 80 microsteps/mm (3200 µsteps/rev ÷ 40mm/rev from 20T GT2 pulley).
