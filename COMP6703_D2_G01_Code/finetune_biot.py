"""Subject-level BIOT adapter experiments on GPU.

Each subject contributes three 10-second windows. The windows are averaged
inside the model before the regression head, so no window from a subject can
leak across a subject-level split.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).parent
sys.path.insert(0, '/tmp/biot_src')
from biot import BIOTEncoder
from evaluate_biot import pooled_windows, load_signal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = '/tmp/biot/EEG-six-datasets-18-channels.ckpt'

def prepare():
    labels = {str(r.id).zfill(3): float(r.label) for r in pd.read_csv(ROOT / 'label.csv').itertuples()}
    val = {str(r.id).zfill(3): float(r.label) for r in pd.read_csv(ROOT / 'val.csv').itertuples()}
    files = {p.name.strip().split('.')[0].zfill(3): p for p in (ROOT / 'EEG').glob('*.mat')}
    ids = sorted(set(labels) | set(val)); ids = [s for s in ids if s in files]
    X = np.stack([pooled_windows(files[s], 18, windows=3) for s in ids])
    y = np.array([{**labels, **val}[s] for s in ids], np.float32)
    is_val = np.array([s in val for s in ids])
    return ids, X, y, is_val

def new_model(mode):
    enc = BIOTEncoder(n_channels=18, n_fft=200, hop_length=100).to(DEVICE)
    enc.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=True)
    for p in enc.parameters():
        if p.is_floating_point():
            p.requires_grad = mode == 'full'
    if mode == 'last_block':
        for p in enc.transformer.layers.layers[-1].parameters(): p.requires_grad = True
    head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1)).to(DEVICE)
    return enc, head

def fit_predict(Xtr, ytr, Xte, mode, epochs=60, seed=7):
    torch.manual_seed(seed); np.random.seed(seed)
    enc, head = new_model(mode)
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-6)
    opt = torch.optim.AdamW([p for p in list(enc.parameters()) + list(head.parameters()) if p.requires_grad], lr=1e-4 if mode != 'head' else 2e-3, weight_decay=1e-3)
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE); yt = torch.from_numpy((ytr - mu) / sd).to(DEVICE)
    enc.train(); head.train()
    for epoch in range(epochs):
        order = torch.randperm(len(Xtr_t), device=DEVICE)
        for ix in order.split(4):
            b = Xtr_t[ix]  # subjects, windows, channels, samples
            emb = enc(b.flatten(0, 1)).reshape(len(ix), b.shape[1], -1).mean(1)
            pred = head(emb).squeeze(-1)
            loss = ((pred - yt[ix]) ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0); opt.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        b = torch.from_numpy(Xte).to(DEVICE)
        pred = head(enc(b.flatten(0, 1)).reshape(len(b), b.shape[1], -1).mean(1)).squeeze(-1).cpu().numpy() * sd + mu
    return np.clip(pred, 0, 40)

def main():
    ids, X, y, is_val = prepare(); tr, va = ~is_val, is_val
    results = []
    for mode, epochs in [('head', 80), ('last_block', 35), ('full', 20)]:
        pred = fit_predict(X[tr], y[tr], X[va], mode, epochs=epochs)
        results.append({'mode': mode, 'epochs': epochs, 'rmse': float(np.sqrt(mean_squared_error(y[va], pred))), 'r2': float(r2_score(y[va], pred)), 'predictions': [{'id': s, 'true': float(t), 'prediction': float(p)} for s, t, p in zip(np.array(ids)[va], y[va], pred)]})
        print(results[-1], flush=True)
    json.dump(results, open(ROOT / 'results/biot_finetune.json', 'w'), indent=2)

if __name__ == '__main__': main()
