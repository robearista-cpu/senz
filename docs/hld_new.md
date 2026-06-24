# High-Level Design (HLD)

# Multi-Accelerometer Finger Tracking Glove Branch

## Document Information

**Project:** Multi-Accelerometer Finger Tracking Glove  
**Branch:** multi-imu-finger-tracking  
**Status:** Draft  
**Version:** 1.0  
**Purpose:** Expand the existing glove architecture with per-finger motion tracking, camera-based position tracking, and force sensing for machine learning dataset generation.

---

# 1. Overview

## Problem Statement

The current glove architecture utilizes a single BNO055 IMU mounted on the hand to estimate overall hand orientation. While this provides global hand pose information, it cannot accurately capture finger articulation or individual finger motion.

The objective of this branch is to create a wearable hand-tracking platform capable of collecting high-quality multimodal datasets for machine learning applications.

The system will combine:

- Wrist orientation tracking
- Per-finger orientation tracking
- Camera-based hand tracking
- Contact and grip sensing

The resulting dataset will support future research in:

- Hand pose estimation
- Motion retargeting
- Gesture recognition
- Contact detection
- Grasp classification
- Human-to-robot hand mapping

---

# 2. Branch Scope

This work shall be developed in a separate branch from the existing BNO055 drift-calibration branch.

The purpose of this separation is to:

- Preserve stability of the existing firmware
- Prevent regression during experimentation
- Allow independent testing and validation
- Reduce integration risk

This branch focuses exclusively on:

- Multi-IMU architecture
- Finger tracking
- Camera integration
- Force sensing
- Dataset generation

---

# 3. System Architecture

## High-Level Architecture

```text
Camera
(MediaPipe Hands)
        │
        ▼

Host Computer
(Data Collection Software)
        ▲
        │ USB Serial

ESP32 / Main Controller
        │
 ┌──────┼───────────────┐
 │      │               │
 ▼      ▼               ▼

BNO055  SPI IMU Array   Force Sensors
(I2C)   (Finger IMUs)   (Velostat)

Wrist    Fingers        Contact Data
```

The architecture intentionally separates:

- Global hand orientation
- Finger articulation
- Hand position
- Contact force estimation

into independent sensing modalities.

---

# 4. Sensor Architecture

## BNO055 Reference IMU

### Purpose

The BNO055 will remain the primary reference sensor for overall hand orientation.

### Interface

- I2C

### Responsibilities

- Wrist orientation
- Hand reference frame
- Global orientation estimate
- Orientation stabilization

### Notes

The BNO055 cannot operate over SPI.

The existing drift-fix and calibration work remains unchanged and will continue to operate independently.

---

## Finger IMU Array

### Purpose

Provide orientation data for individual fingers.

This subsystem is the primary focus of this branch.

### Planned Configuration

Initial implementation:

- Thumb IMUs
- Index IMUs
- Middle IMUs

Future expansion:

- Ring finger IMUs
- Pinky IMUs

Target sensor count:

- Approximately 10 IMUs

### Placement Strategy

Possible layout:

```text
Thumb:
  IMU-1
  IMU-2

Index:
  IMU-3
  IMU-4
  IMU-5

Middle:
  IMU-6
  IMU-7
  IMU-8

Additional:
  IMU-9
  IMU-10
```

Sensors will be attached to finger segments to estimate joint movement and articulation.

### Benefits

Compared to a single wrist IMU:

- Captures finger motion
- Captures articulation
- Provides joint-level information
- Improves gesture recognition
- Enables detailed hand reconstruction

---

# 5. SPI Architecture

## Why SPI

The finger IMU array will operate over SPI.

Advantages:

- Higher bandwidth
- Faster polling rates
- Lower communication latency
- Better scalability for high-frequency sensing

Compared to I2C:

| Feature           | I2C      | SPI    |
| ----------------- | -------- | ------ |
| Speed             | Lower    | Higher |
| Full Duplex       | No       | Yes    |
| Address Conflicts | Possible | None   |
| Scalability       | Moderate | High   |

---

## SPI Expansion Strategy

### Challenge

SPI devices require unique chip-select signals.

With approximately 10 finger IMUs:

- Direct GPIO chip-selects become impractical.

### Solution

Use a chip-select expansion mechanism.

Possible approaches:

- Multiplexer
- Decoder
- Shift register
- GPIO expander

The expansion hardware will:

- Select individual IMUs
- Reduce GPIO usage
- Enable scaling beyond 10 sensors

### Benefits

- Clean wiring
- Reduced pin consumption
- Future expansion support

---

# 6. Camera Tracking System

## Purpose

IMUs provide orientation information but cannot provide absolute hand position.

A camera system will provide:

- Global hand position
- Landmark detection
- Position reference
- Drift correction assistance

---

## Proposed Software

### MediaPipe Hands

MediaPipe Hands will be used for:

- Hand landmark extraction
- Real-time tracking
- Dataset labeling
- Position estimation

---

## Data Produced

Camera system outputs:

- Wrist position
- Finger landmarks
- Palm center
- Hand trajectory

---

## Future Possibilities

Potential upgrades:

- Stereo cameras
- Depth cameras
- Multi-camera tracking
- Full-body integration

---

# 7. Force Sensing Subsystem

## Purpose

Provide contact and grip information during interaction.

This subsystem is intended primarily for machine learning dataset generation.

---

## Sensor Choice

Version 1 will use:

### Velostat-Based Sensors

Reasons:

- Inexpensive
- Readily available
- Lightweight
- Flexible
- Easily integrated into a glove

Industrial load cells were investigated but rejected for Version 1 due to:

- High cost
- Long lead times
- Complex mounting requirements
- Significant wiring complexity

---

## Sensor Construction

Each sensing region consists of:

```text
Conductive Fabric
        │
Velostat Layer
        │
Conductive Fabric
```

Pressure changes resistance.

Resistance is measured using:

- Voltage divider circuits
- ADC sampling
- Signal filtering

---

## Planned Sensor Locations

Initial targets:

- Thumb tip
- Index fingertip
- Middle fingertip
- Ring fingertip
- Pinky fingertip
- Thumb pad
- Index pad
- Middle pad
- Palm center

Approximately:

- 8–10 sensing regions

---

## Data Usage

The force sensing subsystem will provide:

- Contact detection
- Relative grip intensity
- Grasp classification
- Interaction labeling
- ML training features

---

## Limitations

Velostat does not provide calibrated force measurements.

Values should be interpreted as:

- Relative force
- Contact intensity
- Grip strength proxy

not:

- Absolute Newtons

---

## Future Work

Potential future upgrades:

- Miniature load cells
- Button load cells
- Strain gauge sensors
- Absolute force calibration

---

# 8. Data Fusion Strategy

The system will combine information from:

## BNO055

Provides:

- Wrist orientation
- Global reference frame

---

## Finger IMUs

Provide:

- Finger articulation
- Segment orientation
- Joint estimation

---

## Camera Tracking

Provides:

- Hand position
- Landmark references
- Global coordinates

---

## Force Sensors

Provide:

- Contact state
- Relative grip intensity

---

Combined output:

```text
Hand Position
+
Hand Orientation
+
Finger Orientation
+
Contact Information
```

---

# 9. Software Development Objectives

## Objective 1: Multi-IMU Firmware

Develop firmware supporting:

- BNO055
- SPI IMU array
- Sensor scheduling
- Timestamp synchronization
- Serial streaming

### Deliverables

- Sensor drivers
- Polling architecture
- Unified sensor interface
- Data transport layer

---

## Objective 2: Multi-IMU Calibration Program

Develop a standalone calibration application.

### Features

- Sensor discovery
- Per-sensor calibration
- Offset generation
- Validation mode
- Configuration export

### Deliverables

- Calibration tool
- Calibration profile format
- Sensor validation utility

### Success Criteria

- Independent calibration of every IMU
- Persistent calibration profiles
- Reduced drift

---

## Objective 3: Camera Integration Software

Develop software for:

- MediaPipe tracking
- Landmark extraction
- Visualization
- Data export

### Deliverables

- Tracking application
- Landmark recorder
- Synchronization tools

---

## Objective 4: Force Sensor Pipeline

Develop software supporting:

- ADC acquisition
- Signal filtering
- Baseline correction
- Data logging

### Deliverables

- Force sensor driver
- Processing pipeline
- Recording support

---

## Objective 5: Dataset Collection Framework

Create a unified recording system.

### Data Sources

- BNO055
- Finger IMUs
- Camera landmarks
- Force sensors

### Recorded Fields

- Timestamp
- Hand orientation
- Finger orientation
- Hand position
- Force values
- Session metadata

### Deliverables

- Dataset recorder
- CSV exporter
- Binary logger
- Validation utilities

---

## Objective 6: Sensor Fusion Research

Investigate methods for combining:

- BNO055 orientation
- Finger IMUs
- Camera landmarks
- Force sensors

Potential techniques:

- Kalman Filter
- Extended Kalman Filter
- Madgwick Filter
- Learned Fusion Models

---

## Objective 7: Machine Learning Dataset Preparation

Create tooling for:

- Data cleaning
- Timestamp alignment
- Label generation
- Dataset segmentation
- Training export

### Deliverables

- Preprocessing scripts
- Dataset validation tools
- Training-ready datasets

---

# 10. Development Milestones

## Phase 1

- Select SPI IMU
- Design bus architecture
- Design chip-select expansion

---

## Phase 2

- Bring up single finger IMU
- Validate SPI communication
- Validate calibration process

---

## Phase 3

- Scale to full IMU array
- Integrate BNO055 reference frame

---

## Phase 4

- Integrate MediaPipe tracking
- Validate synchronization

---

## Phase 5

- Integrate force sensing regions
- Validate contact detection

---

## Phase 6

- Build dataset collection framework
- Record pilot datasets

---

## Phase 7

- Sensor fusion experiments
- ML dataset generation
- Model development

---

# 11. Success Criteria

The project will be considered successful if:

- 10 IMUs operate reliably
- BNO055 provides stable wrist reference
- Camera tracking remains synchronized
- Force sensors provide usable contact data
- Dataset collection operates without significant data loss
- Multi-modal datasets can be generated for machine learning applications
- The architecture supports future expansion and sensor upgrades
