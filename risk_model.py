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
    # Group equal scores before scoring thresholds: sorting replaces the prior
    # quadratic scan without changing the (recall, precision, threshold) tie rule.
    order=np.argsort(-p,kind='stable');scores=p[order];positive=(y[order]==1)
    ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
    tp=np.cumsum(positive)[ends];precision=tp/(ends+1);recall=tp/(y==1).sum()
    thresholds=scores[ends];valid=precision>=precision_floor
    if not valid.any():raise ValueError('No threshold satisfies the validation precision floor')
    indices=np.flatnonzero(valid);best=indices[np.lexsort((thresholds[valid],precision[valid],recall[valid]))[-1]]
    return {'threshold':float(thresholds[best]),'validation_recall':float(recall[best]),'validation_precision':float(precision[best])}
