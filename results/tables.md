Dataset: 1,125 drives, 18,521 training windows (500 drives); test splits 4,625-4,640 windows (125 drives each).

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

**Roughness** - mean absolute error (median relative error):

| model | seen speeds, new layouts | speed gap 0.95-1.15 m/s | faster 2.0-2.6 m/s | rougher, severity 0.8-1 |
|---|---|---|---|---|
| Random forest, vibration features | 0.19 mm (13%) | 0.20 mm (15%) | 0.37 mm (33%) | 0.64 mm (27%) |
| Random forest, + wheel speed | 0.19 mm (13%) | 0.21 mm (15%) | 0.37 mm (33%) | 0.64 mm (27%) |
| Random forest, per-metre features | 0.18 mm (13%) | 0.18 mm (12%) | 0.24 mm (16%) | 0.64 mm (27%) |
| Linear, vibration features | 0.31 mm (23%) | 0.27 mm (19%) | 0.40 mm (32%) | 0.62 mm (22%) |
| Linear, + wheel speed | 0.31 mm (23%) | 0.27 mm (19%) | 0.41 mm (33%) | 0.62 mm (22%) |
| Linear, per-metre features | 0.31 mm (25%) | 0.31 mm (23%) | 0.32 mm (26%) | 0.59 mm (23%) |
| MLP on spectra + speed | 0.27 mm (19%) | 0.27 mm (19%) | 0.40 mm (34%) | 0.55 mm (23%) |
| MLP on raw signal + speed | 0.30 mm (21%) | 0.31 mm (21%) | 0.62 mm (41%) | 0.69 mm (27%) |
| 1-D CNN on raw signal + speed | 0.19 mm (13%) | 0.18 mm (12%) | 0.53 mm (31%) | 0.50 mm (19%) |

**Roughness on a ground type never seen in training** (random forest, vibration features; trained on the other four types, tested on new layouts of the held-out type):

| held-out type | true mean | error, type seen | error, type never seen |
|---|---|---|---|
| smooth | 0.32 mm | 0.05 mm (12%) | 0.24 mm (66%) |
| gravel | 1.44 mm | 0.20 mm (12%) | 0.37 mm (24%) |
| cobble | 0.97 mm | 0.32 mm (25%) | 0.48 mm (38%) |
| washboard | 1.74 mm | 0.13 mm (6%) | 0.97 mm (62%) |
| rocky | 1.15 mm | 0.27 mm (20%) | 0.33 mm (23%) |
