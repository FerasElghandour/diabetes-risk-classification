"""Tabular neural classification and validation-only threshold selection."""
import numpy as np
import torch
from torch import nn

class RiskMLP(nn.Module):
    def __init__(self,features=21,widths=(64,32),dropout=.1):
        super().__init__()
        if features<1 or not 0<=dropout<1:raise ValueError('Invalid model dimensions or dropout')
        layers=[];previous=features
        for width in widths:
            if width<1:raise ValueError('Hidden widths must be positive')
            layers.extend([nn.Linear(previous,width),nn.SiLU(),nn.Dropout(dropout)]);previous=width
        layers.append(nn.Linear(previous,1));self.layers=nn.Sequential(*layers)
    def forward(self,x):return self.layers(x)

def choose_threshold(labels,probabilities,precision_floor=.72):
    y,p=np.asarray(labels),np.asarray(probabilities,dtype=float)
    if p.ndim!=1 or y.shape!=p.shape or not len(p):raise ValueError('Aligned nonempty vectors required')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isin(y,[0,1]).all():raise ValueError('Binary labels and finite probabilities required')
    if not 0<=precision_floor<=1 or not (y==1).any():raise ValueError('Valid floor and positive labels required')
    candidates=[]
    for threshold in np.unique(np.r_[0.,p,1.]):
        pred=p>=threshold;tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum())
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(y==1).sum()
        if precision>=precision_floor:candidates.append((float(recall),float(precision),float(threshold)))
    if not candidates:raise ValueError('No threshold satisfies the validation precision floor')
    recall,precision,threshold=max(candidates)
    return {'threshold':threshold,'validation_recall':recall,'validation_precision':precision}
