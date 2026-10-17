"""Apply a trained model with its frozen feature order, scaler and threshold.

Research-only CSV inference; this script does not provide a clinical decision.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from train import build_model
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('csv',type=Path);a=p.parse_args()
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);pre=ck.get('preprocessing');selection=ck.get('validation_threshold')
if not pre or not selection:raise ValueError('Use an inference.pt containing fitted preprocessing and validation threshold')
df=pd.read_csv(a.csv);x=df[pre['features']].to_numpy(dtype=float)
if not np.isfinite(x).all():raise ValueError('Finite feature values required')
x=(x-np.array(pre['mean']))/np.array(pre['scale']);m=build_model(ck['config']);m.load_state_dict(ck['model_state'],strict=True);m.eval()
with torch.no_grad():prob=m(torch.from_numpy(x).float()).sigmoid().flatten().numpy()
print(json.dumps({'probabilities':prob.tolist(),'predictions':(prob>=selection['threshold']).astype(int).tolist(),'threshold':selection['threshold'],'scope':'Research classification output; no clinical validity established'}))
