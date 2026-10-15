"""Fit frozen-encoder regression heads and write predictions/metrics."""
from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

def fit(Xtr,ytr,Xv,alpha=10):
    m=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
    m.fit(Xtr,ytr); return np.clip(m.predict(Xv), 0, 40),m

def main():
    z=np.load('features.npz',allow_pickle=True); ids=z['ids'].astype(str); X=z['X']; y=z['y']; val=z['is_val'].astype(bool)
    tr=~val; Xtr,ytr,Xv,yv=X[tr],y[tr],X[val],y[val]
    # Alpha is selected only by leave-one-subject-out CV on training subjects.
    alphas=[0.1,1,10,100,1000]; scores=[]
    for a in alphas:
        cv_pred=cross_val_predict(make_pipeline(StandardScaler(),Ridge(alpha=a)), Xtr, ytr, cv=LeaveOneOut())
        scores.append((a,float(np.sqrt(mean_squared_error(ytr,cv_pred)))))
    alpha=min(scores,key=lambda q:q[1])[0]
    pred,model=fit(Xtr,ytr,Xv,alpha)
    rows=[{'id':i,'true_label':float(t),'prediction':float(p)} for i,t,p in zip(ids[val],yv,pred)]
    metrics={'n_train':int(tr.sum()),'n_validation':int(val.sum()),'alpha':float(alpha), 'rmse':float(np.sqrt(mean_squared_error(yv,pred))) if len(yv)>1 else None,'r2':float(r2_score(yv,pred)) if len(yv)>1 else None,'validation_mean_baseline_rmse':float(np.sqrt(mean_squared_error(yv,np.full_like(yv,ytr.mean())))) if len(yv)>1 else None,'cv_rmse_by_alpha':scores}
    # Ablations: remove relative-power or spatial features (the feature layout is documented in README).
    ab=[]
    for name,mask in [('full',np.ones(X.shape[1],bool)),('no_relative_power',np.r_[np.ones(25,bool),np.zeros(25,bool),np.ones(27,bool)]),('spectral_only',np.r_[np.ones(50,bool),np.zeros(27,bool)])]:
        pp,_=fit(Xtr[:,mask],ytr,Xv[:,mask],alpha)
        ab.append({'ablation':name,'rmse':float(np.sqrt(mean_squared_error(yv,pp))) if len(yv)>1 else None,'r2':float(r2_score(yv,pp)) if len(yv)>1 else None})
    Path('results').mkdir(exist_ok=True); pd.DataFrame(rows).to_csv('results/validation_predictions.csv',index=False); json.dump({'main':metrics,'ablation':ab},open('results/metrics.json','w'),indent=2)
    print(json.dumps({'main':metrics,'ablation':ab},indent=2))
if __name__=='__main__': main()
