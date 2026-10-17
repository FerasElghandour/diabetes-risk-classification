# Reproduction and benchmark scope

The new full-data UCI CDC reference run uses duplicate-feature-group-safe splits, training-only scaling and validation-only threshold selection. The 50,736-row test has86.16% accuracy but only**1.13% positive-class recall** at the selected threshold. Fitted feature order, scaler and threshold accompany the checkpoint.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline source and generated-data training, checkpoint selection and inference reload checks; not a real-data benchmark | None |
| `train` | Configurable training on supplied fixed splits; default budget differs from named published reference protocols | data, config |
| `reference-training` | Exact published fresh-training budgets on identically prepared splits; CPU reference; checkpoint selection uses validation only | data, config |

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


## Device selection

`train.py --device cpu` is the default reference path. `--device mps` explicitly selects an Apple Silicon GPU; `--device cuda` requires a CUDA GPU. Unsupported or unavailable devices fail explicitly. No automatic CPU fallback is enabled. Metrics record the device, runtime and elapsed time. CPU and GPU runs are separate numerical protocols; equal seeds do not imply equal weights or results across devices. Resume requires the checkpoint's original device type and saves its accelerator RNG state.

For a generated-data device integration check, set `TRAIN_TEST_DEVICE=mps` when running `python test_training_pipeline.py`. This checks training and checkpoint reload, not real-data model quality.

## Same-cohort model and threshold diagnostics

The [aggregate comparison](results/baseline-comparison.json) applies fixed logistic-regression and histogram-gradient-boosting baselines to the identical duplicate-group-safe splits, with training-only scaling. Every threshold is selected only on validation data at predeclared precision floors of 0.50, 0.60 and 0.72. This is an exploratory comparison on a previously inspected test cohort, not independent external validation.

| Model | Test ROC-AUC | Average precision | Recall at 0.50 validation floor | Recall at 0.72 validation floor |
|---|---:|---:|---:|---:|
| Released MLP | 0.8310 | 0.4326 | 31.26% | 1.13% |
| Logistic regression | 0.8229 | 0.4029 | 23.92% | 0.03% |
| Histogram gradient boosting | 0.8302 | 0.4301 | 33.31% | 1.92% |

The MLP's low headline recall largely reflects a stringent operating point; it does not dominate the tree baseline. At the 0.72 validation floor, actual test precision is 70.80% for the MLP and 71.20% for boosting: the validation constraint does not guarantee test precision. Lowering the floor trades precision for recall and is not a clinical recommendation. The released model and headline result remain unchanged.

```sh
OMP_NUM_THREADS=2 python benchmark_baselines.py --data data/cdc.npz --checkpoint checkpoints/model.pt --output runs/baselines
```

All model settings are fixed in the script before test evaluation. No individual health records or predictions are distributed.

## Recorded command definitions

### checks

Offline source and generated-data training, checkpoint selection and inference reload checks; not a real-data benchmark

```text
{python} smoke_test.py
{python} test_thresholds.py
{python} test_training_pipeline.py
```

### train

Configurable training on supplied fixed splits; default budget differs from named published reference protocols

```text
{python} train.py --data {input:data} --config {input:config} --output {output}/training
```

### reference-training

Exact published fresh-training budgets on identically prepared splits; CPU reference; checkpoint selection uses validation only

```text
{python} train.py --data {input:data} --config {input:config} --epochs 10 --batch-size 128 --seed 42 --device cpu --output {output}/reference-42 --precision-floor 0.72
```

The observed local runtime is recorded in [reference-environment-macos.json](reference-environment-macos.json). The reproduction driver hashes source files and inputs for each run. Published result records retain their own original runtime and revision; changing the device or budget creates a different experiment.
