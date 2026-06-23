# senz

> A sensor-driven training glove for capturing fine motor hand movement and building datasets for machine learning.

## Overview

**senz** is a wearable glove for tracking fine motor movements of the hand —
each finger and thumb (flexion and extension) along with fast wrist motion. The
end vision is a glove that streams multiple channels of sensor data to a
computer to drive a live hand visualization and to build datasets for machine
learning.

**This repository's current scope is the hardware.** The sensing software, data
pipeline, visualization, and ML work are downstream phases that depend on a
solid physical platform. The near-term effort is dedicated to designing and
building a glove that is comfortable, durable, and instrumented well enough to
produce clean signals later.

## Scope

### In scope (now) — Hardware
- The physical glove: materials, fit, and comfort for long wear.
- Sensor placement and mounting (finger/thumb flexion, wrist motion).
- Wiring, routing, and strain relief.
- Microcontroller selection and board/power integration.
- The physical link to a host computer (connector, cabling, or wireless module).

### Out of scope (for now) — Later phases
- Firmware sensing logic and signal processing.
- Real-time data streaming protocol and host ingestion.
- Visualization of the hand.
- Dataset recording pipeline and ML models.

These remain part of the long-term vision and will get their own milestones once
the hardware platform is stable.

## Hardware Requirements

These constraints guide every hardware decision.

### R1 — Wearability
- Lightweight and comfortable to wear for extended periods.
- Minimal bulk; wiring routed so it does not impede natural hand motion.
- Breathable, durable materials suitable for repeated use.
- Adjustable/secure fit so sensors stay in consistent positions.

### R2 — Sensor Mounting & Coverage
- Mount points for per-finger and thumb flexion/extension sensing.
- A stable mounting location at the wrist for motion sensing.
- Sensors held firmly enough to track both gross and fine movement without
  shifting during use.

### R3 — Electrical & Integration
- Reliable wiring with strain relief at flex points.
- Microcontroller and power integrated without compromising comfort.
- A clean physical link to a host computer (wired connector or wireless module).
- Serviceable construction so sensors and wiring can be repaired or replaced.

## Hardware Milestones

> These are loose guidelines, not a fixed roadmap. They are meant to give a
> rough sense of direction and will shift as the project evolves.

- Design & planning: pick the base glove, sensor types, microcontroller, power,
  and host link; sketch out a bill of materials.
- Sensor mounting: figure out where finger, thumb, and wrist sensors sit and
  make sure they stay put during motion.
- Wiring & routing: route wiring cleanly with strain relief and consolidate
  connections toward the microcontroller.
- Electronics integration: mount the microcontroller and power, connect the
  sensors, and get a physical link to a host computer.
- Comfort & durability: wear-test for long sessions and verify the build
  survives repeated hand motion, iterating as needed.
- Hardware bring-up: confirm sensors produce readable signals and hand off a
  stable platform to the sensing/software phase.

> Sensing, real-time streaming, visualization, and ML work come later, once the
> hardware platform feels solid.

## Proposed Repository Structure

```
senz/
├── hardware/      # Wiring diagrams, BOM, mechanical notes, design files
├── docs/          # Build notes, assembly steps, wear-testing results
│
│   # later phases (created when the hardware is stable):
├── firmware/      # Microcontroller code: sensor reads + host streaming
├── host/          # Host-side ingestion, calibration, visualization
├── data/          # Recorded sessions (datasets)
├── ml/            # Training, evaluation, and inference code
└── sim/           # Simulation + policy for movement regeneration
```

> The hardware-related folders are the current focus; the rest are placeholders
> for the downstream phases.

## Prototype

An 8-hour proof-of-concept build demonstrating 4-finger flexion/extension and
live hand orientation is documented in [`docs/PROTOTYPE.md`](docs/PROTOTYPE.md),
with firmware in `firmware/senz_glove_prototype/` and a live visualization in
`host/live_hand_viz.py` (supports a `--simulate` mode for hardware-free testing).

## Status

Early stage — **hardware development is the active focus.** Sensing, streaming,
visualization, and machine learning are planned future phases that build on the
glove hardware delivered here.
