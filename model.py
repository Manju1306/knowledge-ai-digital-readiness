"""Reproducible knowledge-aware enterprise readiness benchmark.
Synthetic data are for software verification, NOT empirical validation.
Input CSV requires columns FEATURES and independent binary label 'ready'.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['innovation_spend_ratio','digital_skills','data_quality','leadership_support','process_digitization','innovation_collaboration']
RULES = [('data_quality', .35, 'Improve enterprise data quality'),('digital_skills', .40, 'Invest in workforce digital skills'),('leadership_support', .40, 'Strengthen leadership sponsorship')]

def synthetic(n=1200, seed=42):
    rng=np.random.default_rng(seed)
    x=rng.beta(2.5,2.5,(n,len(FEATURES)))
    logit=-3+1.4*x[:,0]+1.1*x[:,1]+1.7*x[:,2]+1.3*x[:,3]+1.1*x[:,4]+.8*x[:,5]
    y=rng.binomial(1,1/(1+np.exp(-logit)))
    frame=pd.DataFrame(x,columns=FEATURES); frame['ready']=y
    return frame

def knowledge_layer(frame, strength=.18):
    """Transparent, predeclared adjustment; NOT trained on outcomes."""
    fail=np.zeros(len(frame),dtype=float)
    for col,threshold,_ in RULES:
        fail+=(frame[col].to_numpy()<threshold).astype(float)
    return fail*strength/len(RULES)

def explain(row):
    return [message for col,threshold,message in RULES if row[col]<threshold]

def metrics(y,prob):
    pred=(prob>=.5).astype(int)
    return {'auroc':float(roc_auc_score(y,prob)), 'average_precision':float(average_precision_score(y,prob)), 'brier':float(brier_score_loss(y,prob)), 'balanced_accuracy':float(balanced_accuracy_score(y,pred))}

def run(df, output, seed=42):
    missing=set(FEATURES+['ready'])-set(df.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    if not set(df['ready'].dropna().unique()).issubset({0,1}) or df['ready'].isna().any(): raise ValueError('ready must be binary with no missing labels')
    if len(df)<100: raise ValueError('At least 100 rows required for demonstration; research validation needs more.')
    if df['ready'].value_counts().min()<10: raise ValueError('At least 10 samples per class required')
    if df[FEATURES].isna().any().any(): raise ValueError('Knowledge rules require complete feature values; preprocess missing data before running.')
    if ((df[FEATURES]<0)|(df[FEATURES]>1)).any().any(): raise ValueError('Features must be scaled to [0,1] with documented measurement.')
    train_idx,test_idx=train_test_split(np.arange(len(df)),test_size=.25,random_state=seed,stratify=df['ready'])
    Xtr,Xte=df.iloc[train_idx][FEATURES],df.iloc[test_idx][FEATURES]
    ytr,yte=df.iloc[train_idx]['ready'],df.iloc[test_idx]['ready']
    base=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(max_iter=1000,random_state=seed))])
    model=CalibratedClassifierCV(base,method='sigmoid',cv=3)
    model.fit(Xtr,ytr)
    p=model.predict_proba(Xte)[:,1]
    adjusted=np.clip(p-knowledge_layer(Xte),0,1)
    results={'dataset_rows':len(df),'test_rows':len(test_idx),'label_definition':'User-provided independent ready label; synthetic demonstration if generated','baseline':metrics(yte,p),'knowledge_adjusted':metrics(yte,adjusted),'rules':RULES,'warning':'Rule thresholds and penalty are illustrative; validate and tune on training/validation data only.'}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(results,indent=2))
    pd.DataFrame({'row_index':test_idx,'observed':yte.to_numpy(),'baseline_probability':p,'knowledge_probability':adjusted,'recommended_actions':['; '.join(explain(r)) or 'No rule flags' for _,r in Xte.iterrows()]}).to_csv(output/'predictions.csv',index=False)
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',help='Enterprise CSV with FEATURES and independently defined ready label');ap.add_argument('--output',default='outputs');ap.add_argument('--seed',type=int,default=42);args=ap.parse_args()
    df=pd.read_csv(args.data) if args.data else synthetic(seed=args.seed)
    print(json.dumps(run(df,args.output,args.seed),indent=2))
