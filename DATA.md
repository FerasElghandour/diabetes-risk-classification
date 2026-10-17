# Input contract

RiskMLP accepts a floating-point N-by-21 feature tensor and returns N-by-1 raw logits. Fit preprocessing only on training records, use BCEWithLogitsLoss for training, and pass validation labels/probabilities to choose_threshold. The core does not provide a trained model or health dataset. Its synthetic checks are not clinical validation.
