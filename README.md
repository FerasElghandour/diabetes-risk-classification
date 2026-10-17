# Diabetes Risk Classification with PyTorch

**Feras Elghandour** · Technical case study

Study the sensitivity–precision trade-off of tabular neural classifiers while preventing duplicate feature vectors from leaking across data splits.

## Implementation

A configurable SiLU tabular MLP and validation-only threshold selection under a precision floor.

```mermaid
flowchart LR
    A[Validated inputs] --> B[Portable implementation]
    B --> C[Evaluation and contract checks]
    C --> D[Inspectable results]
```

## Completed study

The saved SiLU experiment reports test recall 0.8253, precision 0.7145, F1 0.7659, and ROC AUC 0.8235 at threshold 0.47. It records 4,137 true positives, 876 false negatives, 1,653 false positives, and 3,317 true negatives.

These historical figures describe the completed study. This refactored standalone implementation has separate validation and has not been used to recertify the historical benchmark. The public release contains the portable core and focused tests.

## Run locally

Python 3.11 or later.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_test.py
python smoke_test.py
```

Prepare your own inputs as described in [DATA.md](DATA.md). Tests use generated fixtures and do not need credentials or provider calls. [VALIDATION.md](VALIDATION.md) records their scope.

## Practical relevance and limitations

This is a retrospective classification study, not a clinical decision tool. Multiple experiments are reported on the same test partition, so the record should not be treated as a fresh external validation. Thresholds and prevalence may not transfer to another population.

This edition makes the core behavior reusable and inspectable. Historical metrics and synthetic contract tests provide different kinds of evidence; neither establishes client deployment or commercial impact.
