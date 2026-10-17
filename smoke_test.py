import numpy as np
import torch
from risk_model import RiskMLP,choose_threshold
torch.set_num_threads(2);model=RiskMLP();out=model(torch.randn(8,21))
assert out.shape==(8,1)
torch.nn.functional.binary_cross_entropy_with_logits(out,torch.tensor([[0.],[1.]]*4)).backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
selection=choose_threshold([0,0,1,1],[.1,.4,.6,.9],1.)
assert selection['validation_recall']==1 and selection['validation_precision']==1
try:choose_threshold([0,1],[.9,.1],1.)
except ValueError:pass
else:raise AssertionError('Impossible precision floor was accepted')
print('MLP shapes and gradients, validation threshold selection, and infeasibility checks passed.')
