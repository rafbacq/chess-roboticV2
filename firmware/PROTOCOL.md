# Gantry Serial Protocol v2.1

## Transport
- **Interface**: USB CDC serial (Pico native USB)
- **Baud**: 115200
- **Line ending**: `\n` (newline-terminated)
- **Encoding**: ASCII

## Command Format
```
<seq> <CMD> [args...]
```
- `<seq>`: Monotonic uint16 sequence number (host assigns, firmware echoes)
- `<CMD>`: Command name (uppercase)

## Response Format
```
OK <seq> [detail]       — Command accepted
DONE <seq> [detail]     — Async command completed
ERR:<code> <seq>        — Error
EVT:<type> [detail]     — Unsolicited event
```

## Commands

| Command | Args | Requires Homed | Description |
|---------|------|:-:|---|
| `PING` | — | No | Heartbeat/keepalive |
| `STATUS` | — | No | Report position, state, magnet, homed |
| `HOME` | — | No | Begin 3-phase homing sequence |
| `HALT` | — | No | Emergency stop all motors + magnet off |
| `RELAY` | `ON\|OFF` | No | Motor power relay control |
| `MOVE` | `X<mm> Y<mm> Z<mm>` | Yes | Absolute move to position |
| `MAG` | `ON\|OFF` | Yes | Electromagnet control |
| `PICK` | `<sq>` | Yes | Alias for MAG ON |
| `PLACE` | `<sq>` | Yes | Alias for MAG OFF |
| `JOG` | `<axis> <steps>` | Yes | Relative jog by step count |
| `SETSPEED` | `<axis> <steps/s>` | Yes | Set max speed for axis |

## Events (Unsolicited)

| Event | Detail | Trigger |
|-------|--------|---------|
| `EVT:BOOT` | Version string | Power-on |
| `EVT:HOMING` | Axis + phase | During homing |
| `EVT:HOMED` | `ALL` | Homing complete |
| `EVT:HALT` | Reason code | Emergency stop |
| `EVT:POS` | `X<mm> Y<mm> Z<mm>` | Every 500ms while idle/moving |
| `EVT:BTN_STOP` | — | Stop button pressed |
| `EVT:BTN_RESET` | — | Reset button pressed |
| `EVT:ENDSTOP_X/Y/Z` | — | Endstop hit during move |

## Error Codes

| Code | Meaning |
|------|---------|
| `ERR:PARSE` | Command parse failure |
| `ERR:NOT_HOMED` | Motion command before homing |
| `ERR:BUSY` | Already moving |
| `ERR:HALTED` | In HALT or ERROR state |
| `ERR:AXIS` | Invalid axis letter |
| `ERR:UNKNOWN` | Unrecognized command |

## Homing Sequence
1. Z axis: fast approach → endstop → backoff 5mm → slow approach (1mm/s) → endstop → zero
2. X axis: same 3-phase sequence
3. Y axis: same 3-phase sequence
4. All axes zeroed, state → IDLE

## Safety
- **Watchdog**: If no command received for 2s, HALT + relay OFF (kill motor power)
- **Per-motion timeout**: If move exceeds estimated_time × 1.5 + 3s, HALT
- **Endstop during move**: Immediate HALT
- **HALT state**: Magnet forced OFF. Use HOME or button RESET to recover.
- **Motion refused before homing**: All MOVE/JOG/MAG/PICK/PLACE require homed=true

## Example Session
```
→ 1 PING
← OK 1 PONG
→ 2 HOME
← OK 2 HOMING
← EVT:HOMING Z_FAST
← EVT:HOMING Z_BACKOFF
← EVT:HOMING Z_SLOW
← EVT:HOMING X_FAST
...
← EVT:HOMED ALL
→ 3 MOVE X100.0 Y50.0 Z10.0
← OK 3 TO X100.0 Y50.0 Z10.0
← EVT:POS X52.30 Y25.10 Z5.20
← DONE 3 AT X100.00 Y50.00 Z10.00
→ 4 MAG ON
← OK 4 MAG_ON
→ 5 STATUS
← OK 5 x=100.00 y=50.00 z=10.00 state=IDLE mag=1 homed=1
```
