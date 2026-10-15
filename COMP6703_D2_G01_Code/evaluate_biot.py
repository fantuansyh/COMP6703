"""Evaluate public BIOT pretrained encoders with a leakage-safe Ridge probe.

The course recordings use 128 EGI channels while BIOT checkpoints expect 16 or
18 montage channels. We therefore use deterministic contiguous regional pooling
to create the required channels, average three non-overlapping 10-second window
embeddings per subject, and fit only the Ridge head on the 40 training subjects.
"""
from pathlib import Path
import json, sys
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, resample_poly
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, '/tmp/biot_src')
from biot import BIOTEncoder

ROOT = Path(__file__).parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPTS = {
    'BIOT-PREST-16': ('/tmp/biot/EEG-PREST-16-channels.ckpt', 16),
    'BIOT-SHHS+PREST-18': ('/tmp/biot/EEG-SHHS+PREST-18-channels.ckpt', 18),
    'BIOT-six-datasets-18': ('/tmp/biot/EEG-six-datasets-18-channels.ckpt', 18),
}

def load_signal(path):
    d = sio.loadmat(path)
    candidates = [(k, v) for k, v in d.items() if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] in (128, 129)]
    x = max(candidates, key=lambda kv: kv[1].size)[1].astype(np.float32, copy=False)[:128]
    fs = float(np.asarray(d.get('EEGSamplingRate', [[250]])).ravel()[0])
    return x, fs

def pooled_windows(path, n_channels, windows=3):
    x, fs = load_signal(path)
    # Match BIOT's 200 Hz, 10-second input contract.
    sos = butter(4, [0.5, 45], btype='bandpass', fs=fs, output='sos')
    x = sosfiltfilt(sos, x, axis=1)
    if abs(fs - 200) > 1e-3:
        x = resample_poly(x, 200, int(round(fs)), axis=1)
    x = x - np.median(x, axis=0, keepdims=True)
    # Pool all 128 EGI channels into deterministic contiguous regions. np.array_split
    # keeps the two remainder channels instead of silently dropping them for 18ch.
    x = np.stack([x[idx].mean(axis=0) for idx in np.array_split(np.arange(x.shape[0]), n_channels)])
    length = 2000
    starts = np.linspace(0, max(0, x.shape[1] - length), windows, dtype=int)
    out = []
    for st in starts:
        w = x[:, st:st + length]
        w = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-6)
        out.append(w.astype(np.float32))
    return np.stack(out)

@torch.no_grad()
def encode_subject(path, model, n_channels, n_windows=3):
    windows = torch.from_numpy(pooled_windows(path, n_channels, windows=n_windows)).to(DEVICE)
    # One model instance per checkpoint; embeddings are frozen and averaged.
    return model(windows).mean(dim=0).cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=list(CKPTS), choices=list(CKPTS))
    ap.add_argument('--windows', type=int, default=3)
    args = ap.parse_args()
    labels = {str(r.id).zfill(3): float(r.label) for r in pd.read_csv(ROOT / 'label.csv').itertuples()}
    val = {str(r.id).zfill(3): float(r.label) for r in pd.read_csv(ROOT / 'val.csv').itertuples()}
    files = {p.name.strip().split('.')[0].zfill(3): p for p in (ROOT / 'EEG').glob('*.mat')}
    ids = sorted(set(labels) | set(val)); ids = [s for s in ids if s in files]
    all_y = {**labels, **val}; is_val = np.array([s in val for s in ids]); y = np.array([all_y[s] for s in ids])
    rows = []
    for name in args.models:
        checkpoint, nch = CKPTS[name]
        print('encoding', name, flush=True)
        model = BIOTEncoder(n_channels=nch, n_fft=200, hop_length=100).to(DEVICE).eval()
        state = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(state, strict=True)
        X = np.vstack([encode_subject(files[s], model, nch, args.windows) for s in ids])
        np.savez(ROOT / f'results/{name}_embeddings.npz', ids=np.array(ids), X=X, y=y, is_val=is_val)
        tr, va = ~is_val, is_val
        alphas = [1, 10, 100, 1000]
        cv = []
        for a in alphas:
            p = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=a)), X[tr], y[tr], cv=LeaveOneOut())
            cv.append((a, float(np.sqrt(mean_squared_error(y[tr], p)))))
        alpha = min(cv, key=lambda z: z[1])[0]
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[tr], y[tr])
        pred = np.clip(m.predict(X[va]), 0, 40)
        rows.append({'model': name, 'n_channels': nch, 'alpha': alpha,
                     'rmse': float(np.sqrt(mean_squared_error(y[va], pred))),
                     'r2': float(r2_score(y[va], pred)), 'cv_rmse_by_alpha': cv,
                     'predictions': [{'id': s, 'true': float(t), 'prediction': float(p)} for s, t, p in zip(np.array(ids)[va], y[va], pred)]})
    (ROOT / 'results').mkdir(exist_ok=True)
    json.dump(rows, open(ROOT / 'results/biot_comparison.json', 'w'), indent=2)
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()
