"""Fixed-model, same-cohort diagnostics; no test-set hyperparameter tuning."""
import argparse,hashlib,json,time,sys
from pathlib import Path
import numpy as np,torch,sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,precision_score,recall_score,balanced_accuracy_score,confusion_matrix
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from risk_model import choose_threshold
from train import build_model

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Use new output directory')
 digest=hashlib.sha256(a.data.read_bytes()).hexdigest();ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 if ck['data_sha256']!=digest:raise ValueError('Checkpoint/data mismatch')
 with np.load(a.data,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 result={'protocol':'Exploratory same-cohort comparison after the neural result was previously inspected; not untouched external validation','data_sha256':digest,'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'selection':'Models fixed before this comparison; thresholds maximize validation recall subject to each declared precision floor; no test tuning','versions':{'numpy':np.__version__,'torch':str(torch.__version__),'sklearn':sklearn.__version__},'test_prevalence':float(d['y_test'].mean()),'models':{}}
 candidates={'logistic_regression':LogisticRegression(C=1.,max_iter=1000,random_state=42),'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=100,learning_rate=.1,max_leaf_nodes=31,random_state=42)}
 for name in ['released_mlp',*candidates]:
  started=time.perf_counter()
  if name=='released_mlp':
   torch.set_num_threads(2);m=build_model(ck['config']);m.load_state_dict(ck['model_state']);m.eval()
   with torch.no_grad():vp=m(torch.from_numpy(d['x_val']).float()).sigmoid().flatten().numpy();tp=m(torch.from_numpy(d['x_test']).float()).sigmoid().flatten().numpy()
  else:
   m=candidates[name];m.fit(d['x_train'],d['y_train']);vp=m.predict_proba(d['x_val'])[:,1];tp=m.predict_proba(d['x_test'])[:,1]
  y=d['y_test'];row={'roc_auc':float(roc_auc_score(y,tp)),'average_precision':float(average_precision_score(y,tp)),'brier':float(brier_score_loss(y,tp)),'threshold_policies':{},'seconds':time.perf_counter()-started}
  for floor in [.5,.6,.72]:
   try:sel=choose_threshold(d['y_val'],vp,floor)
   except ValueError as exc:row['threshold_policies'][str(floor)]={'unavailable':str(exc)};continue
   pred=tp>=sel['threshold'];row['threshold_policies'][str(floor)]={**sel,'test_precision':float(precision_score(y,pred,zero_division=0)),'test_recall':float(recall_score(y,pred,zero_division=0)),'test_balanced_accuracy':float(balanced_accuracy_score(y,pred)),'test_confusion_matrix':confusion_matrix(y,pred).tolist()}
  result['models'][name]=row;print(json.dumps({name:row}),flush=True)
 a.output.mkdir(parents=True);(a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
