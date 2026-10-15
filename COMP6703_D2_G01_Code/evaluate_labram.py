"""Frozen LaBraM base evaluation on the course EEG split."""
from pathlib import Path
import json, sys
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

sys.path.insert(0, '/tmp/labram')
from modeling_finetune import labram_base_patch200_200

ROOT = Path(__file__).parent

def load_signal(path):
    d = sio.loadmat(path)
    candidates = [(k, v) for k, v in d.items() if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] in (128, 129)]
    x = max(candidates, key=lambda kv: kv[1].size)[1].astype(np.float32, copy=False)[:128]
    fs = float(np.asarray(d.get('EEGSamplingRate', [[250]])).ravel()[0])
    return x, fs

def windows(path, count=3):
    x, fs = load_signal(path)
    sos = butter(4, [0.1, 75], btype='bandpass', fs=fs, output='sos')
    x = sosfiltfilt(sos, x, axis=1)
    if abs(fs - 200) > 1e-3:
        x = resample_poly(x, 200, int(round(fs)), axis=1)
    x = x - np.median(x, axis=0, keepdims=True)
    # 62 deterministic regional channels; all EGI channels are used.
    x = np.stack([x[idx].mean(axis=0) for idx in np.array_split(np.arange(128), 62)])
    length = 1600
    starts = np.linspace(0, max(0, x.shape[1] - length), count, dtype=int)
    out = []
    for st in starts:
        w = x[:, st:st + length]
        w = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-6)
        out.append(w.reshape(62, 8, 200).astype(np.float32))
    return np.stack(out)

@torch.no_grad()
def encode(path, model):
    x = torch.from_numpy(windows(path))
    # Index 0 is CLS; indices 1..62 select 62 channel positions.
    return model(x, input_chans=[0] + list(range(1, 63))).mean(dim=0).numpy()

def main():
    labels = {str(r.id).zfill(3): float(r.label) for r in pd.read_csv(ROOT / 'label.csv').itertuples()}
    val = {str(r.id).zfill(3): float(r.label) for r in pd.read_csv(ROOT / 'val.csv').itertuples()}
    files = {p.name.strip().split('.')[0].zfill(3): p for p in (ROOT / 'EEG').glob('*.mat')}
    ids = sorted(set(labels) | set(val)); ids = [s for s in ids if s in files]
    y = np.array([{**labels, **val}[s] for s in ids]); is_val = np.array([s in val for s in ids])
    model = labram_base_patch200_200(EEG_size=1600, num_classes=0, in_chans=1, out_chans=8, init_values=0.1).eval()
    ckpt = torch.load('/tmp/labram/labram-base.pth', map_location='cpu', weights_only=False)['model']
    prefix = 'student.'
    state = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix) and not k.startswith('student.mask_token') and not k.startswith('student.head')}
    # The pretraining checkpoint names the final normalization `norm`; the
    # fine-tuning class exposes the equivalent layer as `fc_norm`.
    for suffix in ('weight', 'bias'):
        if f'norm.{suffix}' in state:
            state[f'fc_norm.{suffix}'] = state.pop(f'norm.{suffix}')
    model.load_state_dict(state, strict=False)
    X = np.vstack([encode(files[s], model) for s in ids])
    np.savez(ROOT / 'results/labram_embeddings.npz', ids=np.array(ids), X=X, y=y, is_val=is_val)
    tr, va = ~is_val, is_val
    cv = []
    for alpha in [1, 10, 100, 1000, 10000]:
        p = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=alpha)), X[tr], y[tr], cv=LeaveOneOut())
        cv.append((alpha, float(np.sqrt(mean_squared_error(y[tr], p)))))
    alpha = min(cv, key=lambda z: z[1])[0]
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[tr], y[tr])
    pred = np.clip(m.predict(X[va]), 0, 40)
    out = {'model': 'LaBraM-base', 'alpha': alpha, 'rmse': float(np.sqrt(mean_squared_error(y[va], pred))), 'r2': float(r2_score(y[va], pred)), 'cv_rmse_by_alpha': cv, 'predictions': [{'id': s, 'true': float(t), 'prediction': float(p)} for s, t, p in zip(np.array(ids)[va], y[va], pred)]}
    json.dump(out, open(ROOT / 'results/labram_comparison.json', 'w'), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == '__main__': main()
