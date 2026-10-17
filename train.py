"""Train or evaluate the released model using fixed prepared data splits."""
from risk_model import RiskMLP
from training import main

def build_model(c):
    return RiskMLP(features=c["features"],widths=tuple(c["widths"]),dropout=c["dropout"])

if __name__=="__main__":
    main(build_model,{'classes': 2, 'features': 21, 'widths': [64, 32], 'dropout': 0.1},binary=True,tokens=False)
