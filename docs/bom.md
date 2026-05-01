# Bill of Materials — Chess Gantry Robot

## Mechanical

| Item | Qty | Notes |
|------|:---:|-------|
| NEMA17 stepper motor (1.5–2A, 1.8°) | 3 | One per axis (X, Y, Z) |
| A4988 or DRV8825 driver + heatsink | 3 | 1/16 microstepping; DRV8825 if >1.5A needed |
| GT2 timing belt (6mm, open-ended) | 2 | X and Y axes; length per axis travel + 100mm |
| GT2 20-tooth pulley | 2 | For X and Y motor shafts |
| GT2 idler bearing (20T, bore 5mm) | 2 | Belt tensioning for X and Y |
| 8mm linear rod (smooth) OR 8mm linear rail | 6 | 2 per axis (X, Y, Z) |
| LM8UU linear bearing | 6 | 2 per axis |
| T8 lead screw (2mm pitch, 2mm lead) | 1 | Z axis; ~100mm length |
| T8 anti-backlash nut | 1 | For Z lead screw |
| Flexible coupler (5mm to 8mm) | 1 | Motor to Z lead screw |
| Mechanical endstop (NC microswitch) | 3 | One per axis home position |

## Electronics

| Item | Qty | Notes |
|------|:---:|-------|
| Raspberry Pi Pico (RP2040) | 1 | Controller |
| 12V 5A PSU (upsize to 7A if stall >1.5A) | 1 | Main power for steppers |
| 5V 3A buck converter | 1 | Logic power for Pico |
| IRLZ44N N-MOSFET | 1 | Electromagnet driver (logic-level gate) |
| 1N4007 flyback diode | 1 | **MANDATORY** across electromagnet coil |
| 220Ω gate resistor | 1 | MOSFET gate current limiting |
| 10kΩ pulldown resistor | 1 | MOSFET gate-to-source default-off |
| N52 neodymium disk magnet (10×5mm) | 1 | End-effector pickup |
| 5V relay module (SPST) | 1 | Motor power kill switch (safety) |
| 470µF electrolytic capacitor (25V) | 1 | Stepper power rail decoupling |
| 1000µF electrolytic capacitor (10V) | 1 | Pico power rail decoupling |
| Momentary push button (NO) | 2 | STOP and RESET buttons |
| USB cable (micro-B or USB-C, per Pico variant) | 1 | Host communication |
| 24 AWG silicone wire (assorted colors) | 1 lot | Wiring |
| JST-XH connectors (2/3/4 pin) | 1 lot | Stepper and sensor connections |

## Optional

| Item | Qty | Notes |
|------|:---:|-------|
| INA219 current sensor breakout | 1 | Power monitoring (I²C, optional) |
| Magnetic chess piece set OR steel-washer mod kit | 1 | Pieces with steel washers under felt pads |
| USB camera (720p+) | 1 | Overhead board perception (connects to host, NOT Pico) |

## Power Budget

| Consumer | Typical | Peak (stall) |
|----------|:-------:|:------------:|
| NEMA17 X axis | 0.8A | 1.5A |
| NEMA17 Y axis | 0.8A | 1.5A |
| NEMA17 Z axis | 0.5A | 1.0A |
| Electromagnet | 0.3A | 0.5A |
| Pico + logic | 0.1A | 0.2A |
| **Total** | **2.5A** | **4.7A** |

> **Conclusion**: 5A PSU is sufficient. If any axis stalls above 1.5A, upsize to 7A.

## Wiring Safety Notes

1. **Electromagnet MUST go through IRLZ44N MOSFET** with:
   - 220Ω resistor between Pico GPIO and MOSFET gate
   - 10kΩ pulldown between gate and source (default OFF)
   - 1N4007 flyback diode across coil (cathode to +V, anode to drain)
   - **Never drive the coil directly from a GPIO pin**

2. **Stepper drivers** need adequate heatsinks and bulk capacitance on the motor power rail.

3. **Camera connects to HOST (Windows/WSL2 USB), NOT to Pico.** There is no camera data pin on the Pico.

4. **Buck converter is power-only.** No data pin to Pico unless INA219 is installed.
