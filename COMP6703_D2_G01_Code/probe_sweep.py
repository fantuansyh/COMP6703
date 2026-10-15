"""Training-only LOOCV sweep for low-parameter regression adapters."""
from pathlib import Path
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).parent
def candidates():
    out = []
    for a in [1, 10, 100, 1000, 10000, 100000]: out.append((f'ridge_{a}', make_pipeline(StandardScaler(), Ridge(alpha=a))))
    for c in [0.1, 1, 10, 100]: out.append((f'svr_{c}', make_pipeline(StandardScaler(), SVR(C=c, gamma='scale', epsilon=0.5))))
    for n in [2, 5, 10, 15]: out.append((f'pls_{n}', make_pipeline(StandardScaler(), PLSRegression(n_components=n, scale=False, max_iter=1000))))
    for h in [(8,), (16,), (32,)]:
        out.append((f'mlp_{h[0]}', make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=h, alpha=10, learning_rate_init=1e-3, max_iter=1000, early_stopping=False, random_state=7))))
    return out

def evaluate(name, X, y, is_val):
    tr, va = ~is_val, is_val
    rows = []
    for mname, model in candidates():
        p = cross_val_predict(model, X[tr], y[tr], cv=LeaveOneOut())
        rows.append((float(np.sqrt(mean_squared_error(y[tr], p))), mname, model))
    cv_rmse, best_name, best = min(rows, key=lambda x: x[0])
    best.fit(X[tr], y[tr]); pred = np.clip(np.asarray(best.predict(X[va])).ravel(), 0, 40)
    return {'representation': name, 'selected': best_name, 'train_cv_rmse': cv_rmse, 'rmse': float(np.sqrt(mean_squared_error(y[va], pred))), 'r2': float(r2_score(y[va], pred)), 'predictions': pred.tolist(), 'all_cv': [(a, b) for a, b, _ in rows]}

def main():
    outputs = []
    for p in sorted((ROOT / 'results').glob('*_embeddings.npz')):
        z = np.load(p); outputs.append(evaluate(p.stem, z['X'], z['y'], z['is_val']))
    # LaBraM has its own filename and is included above; save a compact report.
    json.dump(outputs, open(ROOT / 'results/probe_sweep.json', 'w'), indent=2)
    print(json.dumps(outputs, indent=2))
if __name__ == '__main__': main()
