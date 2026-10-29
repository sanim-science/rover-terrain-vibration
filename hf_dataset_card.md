---
license: cc-by-4.0
task_categories:
- time-series-classification
- tabular-regression
tags:
- robotics
- terrain-classification
- imu
- accelerometer
- vibration
- distribution-shift
- out-of-distribution
- simulation
- pybullet
pretty_name: Rover Terrain Vibration - IMU benchmark with speed and roughness shift
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: val
    path: data/val.parquet
  - split: test_id
    path: data/test_id.parquet
  - split: test_speed_gap
    path: data/test_speed_gap.parquet
  - split: test_speed_fast
    path: data/test_speed_fast.parquet
  - split: test_rougher
    path: data/test_rougher.parquet
---

# Rover Terrain Vibration

A small, fully simulated benchmark for **terrain recognition from vibration under
distribution shift**. A 1/10-scale car (PyBullet's racecar) drives over five
procedurally generated ground types; a simulated phone-grade IMU records how it
shakes. Each row is one **1-second window at 100 Hz** (accelerometer xyz, gyroscope
xyz, wheel-encoder speed) labelled with the **ground type** and the measured
**roughness in millimetres**.

The point is the test splits: each one changes exactly one thing the model never saw.

| split | what's different from train | use it to measure |
|---|---|---|
| `test_id` | new terrain layouts only | baseline accuracy |
| `test_speed_gap` | speeds 0.95-1.15 m/s (a gap inside the training range) | interpolation in speed |
| `test_speed_fast` | speeds 2.0-2.6 m/s (training tops out at 1.6) | extrapolation in speed |
| `test_rougher` | severity 0.8-1.0 (training tops out at 0.65) | extrapolation in roughness |

Splits are assigned per **drive**, never per window, so overlapping windows from one
drive never leak across splits.

## Quick start

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("<user>/<dataset-name>")
row = ds["train"][0]
x = np.stack([row[c] for c in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "wheel_speed"]], 1)  # (100, 7)
row["terrain"], row["roughness_mm"]
```

## Columns

| column | meaning |
|---|---|
| `acc_x/y/z` | specific force in the car frame (x forward, z up), m/s², reads +9.81 on z at rest |
| `gyro_x/y/z` | angular rate in the car frame, rad/s |
| `wheel_speed` | rear-wheel encoder speed, m/s (differs from true speed when wheels slip) |
| `terrain`, `terrain_id` | smooth / gravel / cobble / washboard / rocky (0-4) |
| `roughness_mm` | RMS ground height under the wheel tracks during this second, wavelengths > 1 m removed |
| `severity` | the terrain generator's roughness knob, 0-1 |
| `speed_cmd`, `speed_true` | commanded and true mean forward speed, m/s |
| `run_id`, `window` | drive identifier and window index within the drive (50 % overlap) |
| `terrain_params` | JSON with the generator parameters for that drive |

Raw 100 Hz recordings of every drive are in `runs/` (one `.npz` per drive, plus
`runs.csv`), if you want other window lengths.

## Baselines

**Naming the ground type** - macro-F1 (chance 0.20), 95 % bootstrap interval over drives:

| model | seen speeds, new layouts | speed gap 0.95-1.15 m/s | faster 2.0-2.6 m/s | rougher, severity 0.8-1 |
|---|---|---|---|---|
| Random forest, vibration features | 0.90 (0.87-0.92) | 0.85 (0.81-0.87) | 0.27 (0.22-0.32) | 0.69 (0.65-0.73) |
| Random forest, + wheel speed | 0.91 (0.88-0.93) | 0.89 (0.86-0.92) | 0.29 (0.22-0.34) | 0.72 (0.67-0.75) |
| Random forest, per-metre features | 0.92 (0.90-0.94) | 0.93 (0.91-0.95) | 0.80 (0.76-0.84) | 0.75 (0.72-0.78) |
| Linear, vibration features | 0.88 (0.85-0.89) | 0.89 (0.87-0.91) | 0.57 (0.51-0.62) | 0.78 (0.74-0.80) |
| Linear, + wheel speed | 0.91 (0.88-0.93) | 0.93 (0.91-0.95) | 0.73 (0.68-0.77) | 0.76 (0.72-0.79) |
| Linear, per-metre features | 0.88 (0.86-0.90) | 0.91 (0.88-0.93) | 0.70 (0.65-0.75) | 0.72 (0.67-0.75) |
| MLP on spectra + speed | 0.89 (0.87-0.91) | 0.87 (0.85-0.88) | 0.57 (0.52-0.62) | 0.77 (0.73-0.80) |
| MLP on raw signal + speed | 0.78 (0.75-0.81) | 0.71 (0.67-0.74) | 0.47 (0.42-0.51) | 0.64 (0.60-0.66) |
| 1-D CNN on raw signal + speed | 0.93 (0.91-0.95) | 0.95 (0.94-0.96) | 0.57 (0.51-0.63) | 0.73 (0.69-0.75) |

Feature sets: vibration statistics; + wheel speed; per-metre (spectra converted to bumps per
metre using the wheel speed). Full roughness results, code and the speed-normalisation
features: <GitHub link>.

**Headline:** every model does well at seen speeds (0.88-0.93), but at faster speeds a random
forest falls to 0.27 and a 1-D CNN to 0.57, while per-metre features keep 0.80. If you have
a method that is robust to speed shift, `test_speed_fast` is the split to beat.

## How it was made, and what we checked

Simulator: PyBullet 3.25, physics at 1000 Hz, rigid contact, wheel collision shapes
replaced by spheres of the same radius. We picked this setup because it was the only
one that **reads ~0 vibration on perfectly flat ground**; the stock cylinder wheels,
soft contact, or physics at 100-400 Hz all invent 1-9 m/s² of fake vibration there,
which a classifier would happily learn. Features also agree within ~6 % between
1000 Hz and 2000 Hz physics. Washboard ridges of spacing L at speed v produce a peak
at v / L Hz (within 4 %), as they should. Full code and checks: <GitHub link>.

## Limitations

Simulation only. The car has no suspension and rigid wheels; real tyres, suspension,
frame resonances, IMU mounting, temperature drift and sensor clipping are not modelled.
Treat this as a controlled benchmark for *how models behave under shift*, not as a
stand-in for real rover data.

## Citation

<report / arXiv link once available>
