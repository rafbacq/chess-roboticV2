# Chess Gantry Robot Firmware

## Overview

Arduino firmware for Raspberry Pi Pico (RP2040) controlling a Cartesian XY+Z stepper gantry with electromagnet pickup for chess piece manipulation.

## Hardware

- 3× NEMA17 steppers via A4988 drivers (1/16 microstepping)
- 3× Mechanical endstops (NC, active-low)
- 1× Electromagnet via IRLZ44N MOSFET (with flyback diode)
- 1× 5V relay for motor power safety kill
- 2× Momentary buttons (STOP, RESET)

## Pin Map

| Function | GPIO | Notes |
|----------|:----:|-------|
| X STEP | GP2 | |
| X DIR | GP3 | |
| X ENABLE | GP4 | Active LOW |
| Y STEP | GP5 | |
| Y DIR | GP6 | |
| Y ENABLE | GP7 | Active LOW |
| Z STEP | GP8 | |
| Z DIR | GP9 | |
| Z ENABLE | GP12 | Active LOW |
| Magnet MOSFET | GP10 | Active HIGH, 220Ω gate resistor + 10kΩ pulldown |
| Relay | GP11 | Active HIGH = power ON |
| MS1 | GP13 | A4988 microstepping |
| MS2 | GP14 | |
| MS3 | GP15 | |
| STOP button | GP21 | Active LOW, internal pullup |
| RESET button | GP22 | Active LOW, internal pullup |
| LED | GP25 | Onboard |
| X Endstop | GP26 | NC switch, internal pullup |
| Y Endstop | GP27 | |
| Z Endstop | GP28 | |

**GP0/GP1**: Reserved for USB serial. NOT used for motors.

## A4988 Microstepping (MS1/MS2/MS3 jumpers)

| MS1 | MS2 | MS3 | Resolution |
|:---:|:---:|:---:|:----------:|
| L | L | L | Full step |
| H | L | L | 1/2 step |
| L | H | L | 1/4 step |
| H | H | L | **1/8 step** |
| H | H | H | **1/16 step** ← Default |

Firmware defaults to 1/16 via software pins. Can also be hardwired.

## Building

```bash
# Install PlatformIO
pip install platformio

# Build
cd firmware
pio run

# Upload (hold BOOTSEL, plug in Pico, then):
pio run --target upload

# Monitor serial
pio device monitor --baud 115200
```

## Protocol

See [PROTOCOL.md](PROTOCOL.md) for full serial command reference.

## Safety Features

1. **Watchdog**: 2s timeout — kills relay (motor power) if no host command
2. **Per-motion timeout**: Aborts if move exceeds estimated time × 1.5
3. **Homing required**: All motion commands rejected until HOME completes
4. **Endstop protection**: Motion aborts on unexpected endstop trigger
5. **STOP button**: Hardware emergency stop, immediate motor halt
6. **Magnet auto-off**: Magnet forced OFF on any HALT condition
