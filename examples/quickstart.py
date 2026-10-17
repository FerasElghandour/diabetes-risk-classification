"""Offline example using the public implementation; see README for evidence scope."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import torch
from risk_model import RiskMLP, choose_threshold
# Generated features exercise an untrained model, not medical prediction.
torch.manual_seed(0)
torch.set_num_threads(2)
model = RiskMLP().eval()
features = torch.zeros(4, 21)
labels = torch.tensor([[0.], [1.], [0.], [1.]])
logits = model(features)
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
loss.backward()
result = {'logits_shape': list(logits.shape), 'loss': loss.item(), 'threshold_example': choose_threshold([0, 0, 1, 1], [.1, .4, .6, .9], 1.)}

import json
print(json.dumps(result, indent=2))

