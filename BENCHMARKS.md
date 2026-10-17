# Reproduction and benchmark scope

The new full-data UCI CDC reference run uses duplicate-feature-group-safe splits, training-only scaling and validation-only threshold selection. The 50,736-row test has86.16% accuracy but only**1.13% positive-class recall** at the selected threshold. Fitted feature order, scaler and threshold accompany the checkpoint.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline behavioral regressions; not a fresh dataset benchmark | None |
| `train` | New reference training protocol on supplied fixed splits; validation-selected checkpoint and held-out metrics | data, config |

Supply named paths with `--input name=/absolute/path`. Exact commands are versioned in [reproduction.json](reproduction.json); direct training commands and preparation contracts are below.

## Evidence boundary

The validation precision floor does not guarantee test precision or useful recall. This new imbalanced-cohort protocol differs from the historical study and establishes no clinical validity.

Keep input acquisition and source credits in [DATA.md](DATA.md). Historical results, new reference runs and small contract fixtures have separate evidential scope.

## Direct commands and inputs

Download [UCI dataset 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), its `data.csv`. ID fields are excluded. Exact duplicate feature groups cannot cross partitions; the scaler is fitted only on training rows.

```sh
python prepare_data.py --csv data.csv --output data/cdc.npz
python train.py --data data/cdc.npz --config data/cdc.config.json --epochs 10 --batch-size 128 --seed 42 --precision-floor 0.72 --output runs/cdc
python predict.py --checkpoint checkpoints/model.pt input_features.csv
```

Inference requires the recorded feature columns, never the target. The 72% validation precision floor is a selection rule, not a test guarantee. Per-row health records and predictions are deliberately not distributed. The released aggregate result exposes the very low positive recall.

## Recorded command definitions

### checks

```text
{python} smoke_test.py
{python} test_thresholds.py
{python} test_training_pipeline.py
```

### train

```text
{python} train.py --data {input:data} --config {input:config} --output {output}/training
```

`{python}` is the current interpreter. `{output}` is the new run directory. `{input:name}` refers to a named absolute input path supplied to the driver. Install declared dependencies in a clean environment before executing. Never execute untrusted serialized models or notebooks.
