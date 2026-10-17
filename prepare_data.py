"""Prepare a supplied binary classification CSV with duplicate-group-safe splits."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,required=True);p.add_argument('--target',default='Diabetes_binary');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Use a new output file')
 df=pd.read_csv(a.csv);y=df.pop(a.target).to_numpy();dropped=[c for c in df if c=='ID' or c.startswith('Unnamed:')];df=df.drop(columns=dropped);x=df.apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
 if not np.isfinite(x).all() or not np.isin(y,[0,1]).all():raise ValueError('Finite features and binary labels required')
 groups=pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy()
 folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(x,y,groups))
 test=folds[0][1];val=folds[1][1];train=np.setdiff1d(np.arange(len(y)),np.r_[test,val])
 if set(groups[train])&set(groups[test]) or set(groups[train])&set(groups[val]) or set(groups[test])&set(groups[val]):raise AssertionError('Duplicate group leakage')
 scaler=StandardScaler().fit(x[train]);data={}
 for name,idx in [('train',train),('val',val),('test',test)]:data['x_'+name]=scaler.transform(x[idx]).astype(np.float32);data['y_'+name]=y[idx].astype(np.int64)
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**data)
 a.output.with_suffix('.config.json').write_text(json.dumps({'classes':2,'features':x.shape[1],'widths':[64,32],'dropout':.1},indent=2)+'\n')
 a.output.with_suffix('.preprocessing.json').write_text(json.dumps({'features':df.columns.tolist(),'excluded_identifier_columns':dropped,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'input_sha256':hashlib.sha256(a.csv.read_bytes()).hexdigest(),'target':a.target,'seed':42,'split_indices':{'train':train.tolist(),'validation':val.tolist(),'test':test.tolist()},'unique_feature_groups':len(set(groups))},indent=2)+'\n')
if __name__=='__main__':main()
