"""Leakage-safe subject-level EEG feature extraction.

The encoder is deliberately deterministic and checkpoint-free so the submission
can be rerun offline.  It follows the frozen-encoder interface used by EEGPT /
LaBraM: each recording is converted to frequency tokens, then a small regression
head is fit using training subjects only.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import welch, butter, sosfiltfilt, iirnotch, filtfilt

BANDS = [(1,4),(4,8),(8,13),(13,30),(30,45)]

def load_signal(path):
    d = sio.loadmat(path)
    candidates = [(k,v) for k,v in d.items() if not k.startswith('__') and isinstance(v,np.ndarray) and v.ndim==2 and v.shape[0] in (128,129)]
    if not candidates: raise ValueError(f'No EEG matrix in {path}')
    x = max(candidates, key=lambda kv: kv[1].size)[1].astype(np.float32, copy=False)
    return x[:128], float(np.asarray(d.get('EEGSamplingRate',[[250]])).ravel()[0])

def feature_one(path, max_seconds=1800):
    x, fs = load_signal(path)
    x = x[:, :int(min(x.shape[1], max_seconds*fs))]
    # Robust common-average reference and inexpensive zero-phase filtering.
    x = x - np.median(x, axis=0, keepdims=True)
    sos = butter(4, [0.5, 45], btype='bandpass', fs=fs, output='sos')
    x = sosfiltfilt(sos, x, axis=1).astype(np.float32, copy=False)
    # Downsampling preserves the 1--45 Hz information and reduces PSD cost.
    step = max(1, int(round(fs/125)))
    x = x[:, ::step]; fs2 = fs/step
    f, p = welch(x, fs=fs2, nperseg=min(1024, x.shape[1]), noverlap=512, axis=1)
    band = np.stack([p[:, (f>=lo)&(f<hi)].mean(axis=1) for lo,hi in BANDS], axis=1)
    logb = np.log10(band + 1e-12)
    rel = band / (band.sum(axis=1, keepdims=True)+1e-12)
    feats=[]
    # Channel distribution statistics retain spatial/topographic information.
    for a in (logb, rel):
        feats += list(a.mean(axis=0)) + list(a.std(axis=0)) + list(np.quantile(a,[.1,.5,.9],axis=0).ravel())
    # Regional asymmetry / gradients (coarse but reproducible across net layouts).
    regions = {'front':slice(0,32),'central':slice(32,80),'posterior':slice(80,128)}
    reg = np.array([logb[v].mean(axis=0) for v in regions.values()])
    feats += list(reg.ravel()) + list((reg[0]-reg[-1])) + list((reg[0]-reg[1]))
    # A few canonical ratios often linked to sleep and stress regulation.
    feats += [float((logb[:,0]-logb[:,2]).mean()), float((logb[:,1]-logb[:,3]).mean())]
    return np.asarray(feats, np.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='.') ; ap.add_argument('--out',default='features.npz')
    args=ap.parse_args(); root=Path(args.data); rows=[]
    labels={str(r.id).zfill(3):float(r.label) for r in pd.read_csv(root/'label.csv').itertuples()}
    val={str(r.id).zfill(3):float(r.label) for r in pd.read_csv(root/'val.csv').itertuples()}
    # Accept the supplied archive's occasional leading-space filenames.
    files={p.name.strip().split('.')[0].zfill(3):p for p in (root/'EEG').glob('*.mat')}
    ids=sorted(set(labels)|set(val)); ids=[s for s in ids if s in files]; X=[]
    for i,sid in enumerate(ids):
        print(f'[{i+1}/{len(ids)}] {sid}', flush=True)
        X.append(feature_one(files[sid]))
    all_y={**labels, **val}
    X=np.vstack(X); np.savez(args.out, ids=np.array(ids), X=X, y=np.array([all_y.get(s,np.nan) for s in ids]), is_val=np.array([s in val for s in ids]))
    print('saved',args.out,X.shape)
if __name__=='__main__': main()
