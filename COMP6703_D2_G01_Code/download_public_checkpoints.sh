#!/usr/bin/env bash
set -euo pipefail
mkdir -p /tmp/biot /tmp/labram
curl -L -o /tmp/biot/EEG-PREST-16-channels.ckpt https://raw.githubusercontent.com/ycq091044/BIOT/main/pretrained-models/EEG-PREST-16-channels.ckpt
curl -L -o /tmp/biot/EEG-SHHS+PREST-18-channels.ckpt 'https://raw.githubusercontent.com/ycq091044/BIOT/main/pretrained-models/EEG-SHHS%2BPREST-18-channels.ckpt'
curl -L -o /tmp/biot/EEG-six-datasets-18-channels.ckpt https://raw.githubusercontent.com/ycq091044/BIOT/main/pretrained-models/EEG-six-datasets-18-channels.ckpt
curl -L -o /tmp/labram/labram-base.pth https://raw.githubusercontent.com/935963004/LaBraM/main/checkpoints/labram-base.pth
echo "Downloaded BIOT and LaBraM checkpoints to /tmp/biot and /tmp/labram"
