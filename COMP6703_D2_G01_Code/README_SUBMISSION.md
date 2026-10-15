# COMP6703 Group Discussion 2

This package implements a leakage-safe, subject-level EEG adaptation study. It evaluates real public BIOT and LaBraM foundation-model checkpoints with a frozen encoder + standardized Ridge head, then tests head-only, last-block, and full BIOT fine-tuning. The best reported method is BIOT six-datasets (18-channel checkpoint) + Ridge.

Public checkpoint sources used in the experiments:

- BIOT six-datasets: `https://raw.githubusercontent.com/ycq091044/BIOT/main/pretrained-models/EEG-six-datasets-18-channels.ckpt`
- BIOT PREST: `https://raw.githubusercontent.com/ycq091044/BIOT/main/pretrained-models/EEG-PREST-16-channels.ckpt`
- BIOT SHHS+PREST: `https://raw.githubusercontent.com/ycq091044/BIOT/main/pretrained-models/EEG-SHHS%2BPREST-18-channels.ckpt`
- LaBraM base: `https://raw.githubusercontent.com/935963004/LaBraM/main/checkpoints/labram-base.pth`

Weights are not included in the ZIP because of size and redistribution concerns. The evaluation scripts expect them under `/tmp/biot/` and `/tmp/labram/` (or can be adapted by changing the checkpoint constants).

## Data layout

Place `label.csv`, `val.csv`, and the `EEG/` folder from the course archive beside these scripts. IDs are matched as zero-padded strings, including filenames with accidental leading spaces.

## Reproduce

```bash
python extract_features.py --data . --out features.npz
python train_and_evaluate.py
```

Outputs include `results/biot_comparison.json`, `results/labram_comparison.json`, `results/biot_finetune.json`, and frozen embedding caches. BIOT uses 0.5--45 Hz filtering, 200 Hz resampling, 18 deterministic regional channels, and three 10-second windows. LaBraM uses 0.1--75 Hz filtering, 200 Hz resampling, 62 regional channels, and three 8-second windows. Ridge alpha is selected by leave-one-subject-out CV on training subjects.

To attempt public checkpoint discovery:

```bash
python download_foundation_checkpoint.py
```

To reproduce the foundation-model runs after downloading checkpoints:

```bash
python evaluate_biot.py --models BIOT-six-datasets-18 --windows 3
python evaluate_labram.py
python finetune_biot.py
python make_slides_final.py
```

## Data integrity note

The working data directory contains 50 EEG recordings: 40 training and 10 validation participants, matching the course split. Validation labels are loaded only for the final metrics table; they are never used to fit preprocessing, select alpha, or train the regression head. EEG recordings and derived participant data are intentionally not redistributed in this code ZIP.
