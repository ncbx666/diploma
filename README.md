# Potato illness forecasting on Kaggle

This project turns the reference notebooks into a small Kaggle-friendly training
script. The original notebooks are kept in the repository as references; the main
entry point is `train.py`.

## 1. Install dependencies

In a Kaggle notebook or a fresh clone, run:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present yet, install the core runtime manually:

```bash
pip install pandas numpy scikit-learn openpyxl matplotlib optuna
```

Optional model families need their own packages, for example `statsmodels` for a
full ARIMA implementation and `pytorch-forecasting`/`lightning` for TFT. The
script records skipped optional artifacts in `skipped_plots.txt` or
`skipped_model.txt` instead of creating fake fallbacks.

## 2. Put the Excel file in place

The expected data file is:

```text
Golitcino72-17d2_CLEAN.xlsx
```

Supported locations:

- repository root after cloning;
- `/kaggle/working/Golitcino72-17d2_CLEAN.xlsx`;
- any Kaggle input dataset path under `/kaggle/input/**/Golitcino72-17d2_CLEAN.xlsx`;
- a custom path via `--data` or `DIPLOMA_EXCEL_FILE_PATH`.

The script does **not** download the dataset from the internet automatically.

## 3. Run one quick model

```bash
python train.py \
  --data Golitcino72-17d2_CLEAN.xlsx \
  --models svm \
  --horizons 2 \
  --window-sizes 7 \
  --tune-trials 0
```

By default outputs go to:

```text
/kaggle/working/outputs
```

For a local test outside Kaggle, use `--output-dir ./outputs`.

## 4. Run all supported models

```bash
python train.py --models all --horizons 2 3 --window-sizes 5 7 9 --top-n 2 --tune-trials 10
```

Feature-engineering experiments from the reference notebooks are always run for
each selected model/horizon/window combination. They are intentionally **not**
CLI parameters, so the benchmark grid remains reproducible:

- `baseline`
- `interaction_fe`
- `temporal_fe`
- `tomek_only`
- `tomek_interaction_fe`
- `tomek_temporal_fe`
- `tomek_interaction_temporal_fe`

The variants come from `notebook08038a32e2 (1).ipynb` and
`potato_illness_forecasting_benchmark (2).ipynb`: manual interaction features,
within-year lag/rolling/trend features, and Tomek Links on the training split
only.

Supported model names in `train.py`:

- `logreg`
- `svm`
- `rf`
- `gru`
- `tft`
- `blitecast`
- `xgboost`
- `catboost`
- `lightgbm`
- `arima`
- `sarima`

`all` expands to the full supported list. Optional models that require unavailable
runtime packages are skipped with an explicit text artifact rather than a fake
substitute.

## 5. Important CLI options

```bash
--models catboost        # selected model names, or --models all
--horizons 2             # one horizon
--horizons 2 3           # several horizons
--window-sizes 7         # one feature window
--window-sizes 5 7 9     # several feature windows
--top-n N                # keep only N best result folders per model, ranked by F1
--tune-trials N          # number of Optuna trials; use 0 for a fast smoke test
--output-dir PATH        # default: /kaggle/working/outputs
```

F1 is the primary ranking criterion. Result folders include the model name and
feature-engineering variant plus F1 score as a percentage, for example
`outputs/svm_temporal_fe_62/` for F1 ≈ 0.62. If a name already exists, a safe
suffix is added.

## 6. Outputs and zip files

Each experiment writes a result folder with artifacts such as:

- `metrics.csv`
- `predictions.csv` with required `y_pred_binary`
- `classification_report.csv`
- `classification_report.png`
- `config_used.yaml`
- `train_val_test_years.json`
- plot files when they are applicable
- `skipped_plots.txt` when a plot is not applicable or plotting dependencies are unavailable

After each experiment, the script immediately creates a zip archive in:

```text
/kaggle/working/outputs/zips/
```

Without `--upload-dataset`, zip files are only local. The script prints a warning
because files in `/kaggle/working` can disappear after the Kaggle runtime stops.
Missing Kaggle CLI credentials do not break a normal local-only run.

## 7. Persist outputs in a Kaggle Dataset

To additionally copy each zip to the persistent dataset staging directory and run
Kaggle Dataset versioning:

```bash
python train.py \
  --models svm \
  --horizons 2 \
  --window-sizes 7 \
  --upload-dataset \
  --dataset-slug username/dataset-name
```

With upload enabled, the script copies zips to:

```text
/kaggle/working/outputs/persistent_dataset/
```

It also creates or updates `dataset-metadata.json` using the slug from
`--dataset-slug`, then runs:

```bash
kaggle datasets version -p /kaggle/working/outputs/persistent_dataset/ -m "new experiment results"
```

Upload is retried 3 times. After 3 failed attempts, the error is not hidden.

## 8. Smoke test used during development

```bash
python train.py --models svm --horizons 2 --window-sizes 7 --top-n 1 --tune-trials 0 --output-dir /tmp/diploma-smoke
```

Expected evidence: an experiment folder plus a zip in `/tmp/diploma-smoke/zips/`.
