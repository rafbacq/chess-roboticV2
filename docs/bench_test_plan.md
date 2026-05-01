# Bench Test Plan

## Hardware Tests

| # | Test | Procedure | Pass Criteria |
|---|------|-----------|---------------|
| 1 | Power budget | Measure stall current per axis with ammeter | <5A peak, <3A sustained |
| 2 | Pin assignment | Send PING while motors run; verify clean serial | No garbled bytes, no UART conflict |
| 3 | Homing repeatability | 10× HOME from random positions, measure final position | ±0.5mm across all runs |
| 4 | Square accuracy | Move a1→h8→a1 × 100 cycles, measure final position | <2mm final error, no accumulated drift |
| 5 | Magnet hold | Pick piece → traverse full board → release × 50 | Zero drops during traverse, clean release every time |
| 6 | STOP latency | Press STOP button mid-move, measure time to motor halt | <100ms from button press to zero velocity |
| 7 | Thermal | 30 minutes continuous motion (random squares) | Drivers <80°C, motors <60°C (IR thermometer) |
| 8 | Calibration | 5× run full calibration routine, compare results | Std dev <1mm per corner across runs |
| 9 | Watchdog | Disconnect USB mid-move, verify relay kills power | Motors stop within 2s of disconnect |
| 10 | Endstop safety | Trigger endstop during normal move | Motion aborts immediately, state → HALT |

## Perception Tests

| # | Test | Procedure | Pass Criteria |
|---|------|-----------|---------------|
| 11 | Real-image accuracy | 150-photo stratified test set | macro-F1 ≥ 0.92 |
| 12 | OOD robustness | 50 adversarial photos (low light, glare, hands) | macro-F1 ≥ 0.75 |
| 13 | Inference latency | Batch 64 squares through classifier on RTX 4070 Ti | ≤30ms per board (GPU), ≤200ms (CPU) |
| 14 | Per-class recall | Check each of 13 classes on real test set | Every class ≥ 90% recall |
| 15 | Calibration (ECE) | Temperature scaling on val set | ECE ≤ 2% post-calibration |

## End-to-End Tests

| # | Test | Procedure | Pass Criteria |
|---|------|-----------|---------------|
| 16 | E2E sim game | Full game vs Stockfish-1, sim gantry, 20 moves | Completes without exception |
| 17 | E2E real game | 10 games vs Stockfish-5, real hardware | Zero failures, <10s per move avg |
| 18 | Soak test | 100-game simulated tournament | No memory leaks, no file handle leaks, no race conditions |

## Procedure Notes

- All measurements recorded in `benchmarks/RESULTS.md`
- Each test includes date, firmware version, software git SHA
- Thermal tests use contactless IR thermometer
- Position accuracy tests use dial indicator or camera measurement
- Soak test runs weekly via scheduled job
