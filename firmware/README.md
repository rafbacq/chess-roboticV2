# Firmware — Chess Gantry Robot v2.0

## Architecture

Stepper-based XYZ Cartesian gantry with electromagnet pick-up.

```
           Y axis
           ↑
           |
    [Endstop]──────────────────[Y Motor]
           |                        |
           |     Chess Board        |
           |     (300×300mm)        |
           |                        |
    [Endstop]──────────────────────
           └──→ X axis
          [X Motor]
          [Endstop]

    Z axis: vertical on the Y carriage, lifts the electromagnet head.
```

## Bill of Materials

| Qty | Component | Purpose | Notes |
|-----|-----------|---------|-------|
| 3 | NEMA17 stepper motor (1.5–2A) | XYZ axes | 1.8°/step, bipolar |
| 3 | A4988 or DRV8825 driver module | Stepper control | Heat sink required |
| 1 | Raspberry Pi Pico (RP2040) | Controller | USB serial |
| 3 | GT2 timing belt + 20T pulley | Motion transmission | 2mm pitch |
| 6 | 8mm smooth rod + LM8UU bearing | Linear guides | Or MGN12 linear rails |
| 3 | Mechanical endstop (NC) | Homing reference | Microswitch type |
| 1 | N52 neodymium disk magnet | Piece pickup | ~12mm dia × 3mm |
| 1 | IRLZ44N N-channel MOSFET | Magnet switching | Logic-level gate |
| 1 | 1N4007 rectifier diode | Flyback protection | Across magnet coil |
| 1 | 12V/5A power supply | Motor power | Separate from Pico USB |
| ~ | Steel washers or M3 screws | Piece modification | One per chess piece |
| 1 | Felt pads (adhesive) | Piece base | Covers embedded steel |

## Pin Map (RP2040)

```
GP0, GP1  — RESERVED (USB UART, do not use)
GP2, GP3  — X stepper STEP, DIR
GP4, GP5  — Y stepper STEP, DIR
GP6, GP7  — Z stepper STEP, DIR
GP8, GP9  — Spare (STEP, DIR) — disabled, future use
GP10      — X endstop (NC, INPUT_PULLUP)
GP11      — Y endstop
GP12      — Z endstop
GP13      — Electromagnet MOSFET gate
GP14-16   — Microstepping select (MS1/MS2/MS3)
GP25      — Onboard LED (status indicator)
```

## Building

Requires [PlatformIO](https://platformio.org/):

```bash
cd firmware/
pio run                # Compile
pio run -t upload      # Flash via USB
pio device monitor     # Open serial monitor
```

### DRV8825 vs A4988

By default, the firmware configures A4988 microstepping pins.
To use DRV8825, add `-D USE_DRV8825` in `platformio.ini` build_flags.

## Microstepping

Default: **1/16 microstepping** → 3200 µsteps/revolution.

With a 20-tooth GT2 pulley (40mm/rev):
- **80 microsteps per mm**
- Theoretical resolution: **0.0125mm (12.5 µm)**

## Safety Features

1. **Watchdog**: 2-second timeout. If no command received, auto-HALT.
2. **Endstop polling**: Every step during motion. Hard stop on trigger.
3. **Magnet auto-off**: On HALT or watchdog reset.
4. **Homing required**: No motion commands accepted until HOME completes.
5. **Travel limits**: Software clamping to max XYZ dimensions.
6. **Acceleration ramping**: AccelStepper prevents instant starts/stops.

## Wiring the Electromagnet

```
         Pico GP13 ──── [IRLZ44N Gate]
                         │
    +12V ────── [N52 Magnet Coil] ──── [IRLZ44N Drain]
         │                                    │
         └───── [1N4007 ←──────────────┘      │
                (cathode to +12V)              │
                                          [IRLZ44N Source] ──── GND
```

The 1N4007 flyback diode (cathode to +12V, anode to drain) clamps
back-EMF when the MOSFET turns off, preventing voltage spikes.

## Chess Piece Modification

Each chess piece needs a small ferrous target:
- Drill a shallow 3mm hole in the base
- Epoxy a steel washer (M3×8mm OD) or small screw
- Cover with adhesive felt pad

Or: purchase a magnetic tournament chess set with pre-magnetized bases.

## Serial Protocol

See [PROTOCOL.md](./PROTOCOL.md) for the complete serial command reference.
