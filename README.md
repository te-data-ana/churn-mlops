# Churn MLOps

A local-first binary-classification MLOps project, with customer churn as its
included use case. It demonstrates data preparation, time-based evaluation,
model training and promotion, batch and HTTP inference, and model, data, and
service monitoring.

The project uses scikit-learn pipelines for feature engineering and prediction,
MLflow for experiments and the model registry, DVC for reproducible data and
evaluation pipelines, and FastAPI for serving.

## Quick start

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

Install the project and its development dependencies, then check the CLI.
Include the notebook group to run the included analysis notebooks:

```bash
uv sync --group notebooks
uv run churn-mlops --help
```

Run the test suite:

```bash
uv run pytest
```

Training requires prepared files at
`data/splits/{split_name}_train.parquet` and
`data/splits/{split_name}_test.parquet`. Follow the data preparation and
training workflows below to create them. The repository does not include the
raw CSV contents; see [DVC experiments](#reproduce-an-out-of-time-experiment)
for how to obtain and track those local inputs.

## Workflows

### Prepare data and create a time-based split

Raw input CSVs belong in `data/raw`. The helper below assigns dates from the
requested range to rows and writes an enriched CSV in `data/raw`. Prepare both
training and inference cohorts so the partitioned dataset covers the full
out-of-time window:

```bash
uv run python -m churn_mlops.data.manipulation \
  --use_case churn \
  --input_csv customer_churn_dataset-training.csv \
  --output_csv training.csv \
  --start_date 2025-01-01 \
  --end_date 2025-12-01

uv run python -m churn_mlops.data.manipulation \
  --use_case churn \
  --input_csv customer_churn_dataset-inference.csv \
  --output_csv inference.csv \
  --start_date 2026-01-01 \
  --end_date 2026-06-01
```

`prepare-data` validates a CSV and writes it as a month-partitioned Parquet
dataset under `data/partitioned`. The input argument is a filename under
`data/raw`. Prepare both enriched files:

```bash
uv run churn-mlops prepare-data --input_csv training.csv --use_case churn
uv run churn-mlops prepare-data --input_csv inference.csv --use_case churn
```

Create train and test files from an inclusive date range. Rows before
`split_date` go to training; rows on or after it go to test. The selected use
case supplies the timestamp column.

```bash
uv run churn-mlops create-split \
  --dataset_name partitioned \
  --use_case churn \
  --start_date 2025-04-01 \
  --split_date 2026-04-01 \
  --end_date 2026-06-30 \
  --split_name oot-26Q2
```

This creates `oot-26Q2_train.parquet`, `oot-26Q2_test.parquet`, and split
metadata in `data/splits`.

### Reproduce an out-of-time experiment

The root [dvc.yaml](dvc.yaml) defines a pipeline that adds the selected use
case's time column, prepares partitioned data, creates a time split, trains a
model, then scores the held-out cohort. The top-level `use_case` parameter
selects a registered contract for preparation, splitting, training, and
scoring. Run-time training options are passed as CLI arguments by the DVC
stage. DVC tracks stage inputs and outputs; MLflow stores runs, metrics,
model artifacts, and registered versions.

Optuna tuning is disabled by default in the DVC pipeline; opt in for a run with
`training.tune_model=true`.

The `data/manifests` and `data/predictions` directories are persistent DVC
outputs. Manifest files are named by `{use_case}-{classifier}-{split}`; the
prediction dataset path uses `{use_case}_{classifier}_{split}`. Repeating a
pipeline combination reuses its output paths; it does not create a separate
archive for each run.

The checked-in [params.yaml](params.yaml) selects the `churn` use case and
configures `oot-26Q2`: training from 2025-04-01 through 2026-03-31 and testing
from 2026-04-01 through 2026-06-30, using the `hgb` classifier. Date bounds
are inclusive. Provide the configured training and inference CSVs under
`data/raw` before running the pipeline:

```bash
uv run dvc repro
uv run dvc status
```

To add or update the local source files tracked by DVC, run:

```bash
uv run dvc add \
  data/raw/customer_churn_dataset-training.csv \
  data/raw/customer_churn_dataset-inference.csv
```

Commit the generated `.dvc` pointers, not the CSV contents.

To run another quarter/classifier combination as a named DVC experiment:

```bash
uv run dvc exp run --name churn-12m-oot \
  --set-param use_case=churn \
  --set-param model.classifier=lr \
  --set-param split.name=oot-26Q2 \
  --set-param 'split.start_date="2025-04-01"' \
  --set-param 'split.split_date="2026-04-01"' \
  --set-param 'split.end_date="2026-06-30"' \
  --set-param calibration.method=isotonic \
  --set-param training.n_jobs=4 \
  --set-param training.tune_model=true \
  --set-param model.register_model=true
```

The `use_case` parameter (set in `params.yaml`) selects the registered contract
and is passed as the required `--use_case` argument to training. Training
derives the MLflow experiment name as `{use_case}-{split_name}` and registry
model name as `{use_case}-model`. A registered model receives no alias until it
is promoted; successful promotion assigns `champion`.
To run another registered use case, also update the corresponding input CSVs
and date ranges in `dvc.yaml`.

`model.classifier` selects `src/config/{classifier}.yaml`, so each classifier
alias used by this pipeline must have a matching config file.

The `calibration.method` DVC parameter is passed directly to the training
command as `--calibration_method`; its default is `none`.

`training.n_jobs` is passed as `--n_jobs` and controls the total CPU job budget:
use a positive integer or `-1` for all available CPUs, the default is `1`. Only
the outermost operation in each training phase uses this budget, nested
estimators use only one job to avoid oversubscription.

`training.tune_model` is also passed as a run-time option and defaults to `false`.
Set it to `true` to enable Optuna tuning using the budgets in the selected
classifier YAML.

Registration and promotion default to disabled on direct training calls. The DVC
`model.register_model` parameter enables registration for this pipeline run.
Promotion remains disabled by the stage's `--no-promote-model` argument. The
pipeline scores the logged MLflow model artifact, using the exact registered
version from the manifest when registration is enabled.

The source CSVs are represented in Git by `.dvc` pointer files, not by their
contents. This repository has no DVC remote, so a fresh clone must be supplied
with the CSVs locally; `dvc pull` cannot retrieve them. Likewise, the SQLite
MLflow database and artifact directory are local. Cross-machine reproduction
requires sharing both DVC data and MLflow tracking/artifacts.

### Train and evaluate a model

Training reads prepared Parquet files from `data/splits`, builds and fits the
configured pipeline, evaluates it on the test split, and logs the run to
MLflow.

```bash
uv run churn-mlops train \
  --use_case churn \
  --config rf.yaml \
  --split_name oot-26Q2
```

Configuration filenames resolve under `src/config` by default. The default
split name is `default`. The training configuration controls the classifier,
preprocessing, calibration fold/bin counts, threshold selection, and tuning
search parameters.

Run-time arguments control calibration method, CPU job budget and whether
tuning, model registration and promotion is executed. The latter options
are disabled by default. The required `--use_case` selects the data contract.
Training derives its experiment name from the use case and split, and its
registry model name from the use case. The selected use case's
`feature_columns` define model inputs and must be present in both splits;
other validated columns remain available in batch prediction outputs. See the
[model configuration and tracking reference](#model-configuration-and-tracking-reference)
for configuration details and model behavior.

Python callers can set the same per-run options directly. They default to
- `calibration_method="none"`,
- `n_jobs=1`,
- `tune_model=False`,
- `register_model=False`,
- `promote_model=False`.

```python
from churn_mlops.train import run_training_job

run_training_job(
    use_case="churn",
    config_file="rf.yaml",
    split_name="oot-26Q2",
    register_model=True,
    promote_model=False,
    calibration_method="sigmoid",
    n_jobs=4,
)
```

#### Tune hyperparameters

Optuna tuning is disabled by default (`tune_model=False`). To enable it,
choose a classifier with a catalog search space, set `tune_model=True` in a
Python call (or pass `--tune-model` in the CLI) and configure the search
budget in its YAML:

```yaml
tuning:
  n_trials: 64
  n_splits: 4
  metric: roc_auc
  random_state: 26
```

Tuning uses deterministic expanding-window cross-validation across the full
training split. Rows with the same timestamp stay in the same fold. The
selected use case's time column must be present in the training split, and
every fold must contain both target classes. The out-of-time test data is held
out from hyperparameter tuning.

### Generate batch predictions

Batch prediction validates input data and loads the selected model. Use a
training manifest, an explicit model name and version, or set `MODEL_NAME` to
select a registered model and its configured alias. A training manifest
identifies the logged MLflow model and prediction threshold, whether or not
that model is registered.

For CSV input, the input filename is resolved under `data/raw` and the output
is written under `output`:

```bash
uv run churn-mlops batch-predict \
  --use_case churn \
  --input_csv inference.csv \
  --output_csv predictions_inference.csv \
  --index_col customerid
```

For a partitioned dataset, input and output dataset names are relative to
`DATA_DIR`. Date bounds must be provided together or omitted together; a
date-only end bound includes that calendar day. Existing output partitions for
the months in the result are replaced.

```bash
uv run churn-mlops batch-predict \
  --use_case churn \
  --input_dataset partitioned \
  --start_date 2026-06-01 \
  --end_date 2026-06-30 \
  --output_dataset batch_predictions
```

The partition timestamp defaults to the selected use case's time column; the
default output dataset is `batch_predictions`. The result includes the input
features and
`predicted_probability`, `predicted_class`, `threshold`, and `model_version`.

To score a specific model version, pass both its registered name and version.
Alternatively, pass a manifest path relative to `DATA_DIR`:

```bash
uv run churn-mlops batch-predict \
  --use_case churn \
  --input_dataset splits/oot-26Q2_test.parquet \
  --output_dataset predictions/rf_oot-26Q2 \
  --model_manifest manifests/churn-rf-oot-26Q2.json
```

The manifest option cannot be combined with `--model_name` or
`--model_version`.

### Serve predictions over HTTP

Start the FastAPI service locally. Set `MODEL_NAME` to a registered model name
derived from its use case name, or use `--model_manifest` to select the exact
model recorded by a training manifest (including unregistered runs). The
selected model must have been trained for the selected use case; mismatches
prevent readiness:

```bash
MODEL_NAME=churn-model \
  uv run churn-mlops serve --use_case churn --host 127.0.0.1 --port 8000 --reload
uv run churn-mlops serve --use_case churn --model_manifest manifests/churn-lr-oot-26Q2.json
```

The service loads the selected model once per process. Its
`/health` endpoint is a liveness check; `/ready` reports whether a model loaded
successfully. If model loading fails, the process remains live, but `/ready`
and `/predict` return `503` until it is restarted with a valid model
configuration.

```bash
curl http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
```

Send a record to `/predict`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42,
    "tenure": 18,
    "usage_frequency": 12,
    "support_calls": 2,
    "payment_delay": 0,
    "last_interaction": 7,
    "total_spend": 1250.50,
    "gender": "Female",
    "subscription_type": "Standard",
    "contract_length": "Annual"
  }'
```

An optional `timestamp` query parameter can override the inference time. The
response and prediction/error logs name the timestamp field using the selected
use case's `time_column`; the generated API schema reflects that field name.

The API uses the `MODEL_NAME` and `MODEL_ALIAS` settings (or an explicit model
version) to load a registered model. The interactive API reference is available
at <http://127.0.0.1:8000/docs>.

For sample-based API checks, the helper reads and samples the CSV given by
`--input_csv` (resolved in `data/raw`), sends each row to the service, and
writes predictions with the input features to the configured output directory:

```bash
uv run python -m churn_mlops.serving.serve_samples \
  --use_case churn \
  --input_csv inference.csv \
  --model_name churn-model \
  --model_version 1 \
  --sample_size 2000 \
  --random_state 42 \
  --timestamp 2026-09-01 \
  --drop_columns churn reference_date
```

### Monitor model and service behavior

The `monitor` command compares labeled reference data with scored batch data or
API prediction logs. It calculates monthly metrics for model versions present
in both cohorts. Prefer a held-out or out-of-time reference cohort over
in-sample training predictions.

Monitor scored CSV files:

```bash
uv run churn-mlops monitor \
  --use_case churn \
  --reference_csv output/predictions_training.csv \
  --analysis_csv output/predictions_inference.csv
```

Monitor the configured API prediction and error logs:

```bash
uv run churn-mlops monitor \
  --use_case churn \
  --reference_csv output/predictions_training.csv \
  --api
```

Reports are written under `output` by default. Use `--output_dir` to choose
another location. API logs can be selected explicitly with `--prediction_log`
and `--error_log`.

Monitoring can also use already-scored partitioned datasets under `DATA_DIR`:

```bash
uv run churn-mlops monitor \
  --use_case churn \
  --reference_dataset train_predictions \
  --analysis_dataset api_predictions \
  --error_dataset api_errors \
  --source_reference batch \
  --source_analysis api \
  --output_dir output/monitoring
```

Dataset names are relative to `DATA_DIR`. Supply both reference and analysis
datasets, and set each source to `batch` or `api`; the default analysis source
is `api`. Batch datasets and API logs use the selected use case's time column.
The datasets must already contain prediction columns, `model_version`, and that
time column. Reference labels are needed for measured performance; model
features are needed for drift analysis. Error datasets are optional.

Monitoring reports include prediction counts, average probability, and
predicted-positive rate. Depending on available data, they also include:

- measured precision, recall, ROC AUC, and observed positive rate when labels
  are available;
- NannyML confidence-based performance estimates when analysis labels are
  unavailable and the reference data is labeled;
- feature drift alert counts when the required features are present and the
  reference and analysis periods do not overlap; and
- API request counts, error rates, and successful/failed request latency
  percentiles for API-based analysis.

The CSV report is named `monitoring_summary_batch.csv` or
`monitoring_summary_api.csv`, depending on the analysis source.

For repeatable cohort selection and MLflow tracking, the separate
[monitoring pipeline](monitoring_pipeline/dvc.yaml) uses settings from
[monitoring_pipeline/params.yaml](monitoring_pipeline/params.yaml):

```bash
uv run dvc repro monitoring_pipeline/dvc.yaml:monitor
```

This writes a scenario-specific report under `data/monitoring` and logs the
report, cohort names, and model version as an MLflow monitoring run. Choose
non-overlapping time periods when drift metrics are required.

API prediction and error logs can be converted into month-partitioned Parquet
datasets. Select the use case so its time column is used for partitioning. By
default, these commands read the configured JSONL logs and write to
`data/api_predictions` and `data/api_errors`:

```bash
uv run churn-mlops ingest-prediction-logs --use_case churn
uv run churn-mlops ingest-error-logs --use_case churn
```

Re-ingestion replaces existing partitions for the months represented in the
logs. Use `--dataset_name` or `--timestamp_column` to override the destination
dataset or partition timestamp.


## Use case specifications

Use-case-specific behavior lives under [use_cases](src/churn_mlops/use_cases):
each use case defines its target and time columns, feature columns, data
validator, feature-builder factory, and optional HTTP request schema. Shared
training, batch prediction, API, and monitoring code consumes this contract.

To add another classification problem:

1. Add a use_case package with its data validator and (when needed) feature
   transformer and Pydantic request schema. Use the shared
   [identity feature transformer](src/churn_mlops/models/features.py) when no
   domain-specific feature engineering is required.
2. Register a `UseCase` in [registry.py](src/churn_mlops/use_cases/registry.py),
   providing its target, time and feature columns, validator, transformer
   factory, and optional request schema.
3. Add a training YAML with model settings; pass the registered use case
   explicitly when invoking training.
4. Select the use case explicitly with `--use_case` for data preparation,
   batch prediction, monitoring, serving, and API log ingestion. Dataset and
   API log timestamps use the selected use case's time column.

Training configurations are not substitutes for input validation or HTTP
schemas: each use case owns those contracts. Retrain models after changing
use_case feature logic.

### Data contract for churn use case

Training and inference data use the following customer features:

| Field | Type and constraints |
| --- | --- |
| `age` | Integer from 18 to 115 |
| `tenure` | Non-negative integer |
| `usage_frequency` | Non-negative integer |
| `support_calls` | Non-negative integer |
| `payment_delay` | Non-negative integer |
| `last_interaction` | Non-negative integer |
| `total_spend` | Non-negative number |
| `gender` | `Female` or `Male` |
| `subscription_type` | `Basic`, `Standard`, or `Premium` |
| `contract_length` | `Monthly`, `Quarterly`, or `Annual` |

For this churn use case, training data also requires a binary `churn` target
(`0` or `1`). Depending on the workflow, data may also contain `customerid`
and `reference_date`. CSV loading normalizes column names; `customerid` can be
selected as the CSV index.

The churn data contract is validated with Pandera:
[churn data schema](src/churn_mlops/use_cases/churn/schema.py)

Its HTTP request features are validated with Pydantic:
[churn request schema](src/churn_mlops/use_cases/churn/serving.py).


## Model configuration and tracking reference

### Configuration and classifiers

Training configurations are YAML files under [src/config](src/config). They
define classifier parameters, preprocessing strategies, calibration and
threshold-selection strategy, tuning parameters, and promotion thresholds and
weights.

Use case, calibration method, job budget, tuning, registration and promotion
need to be explicitly selected/enabled for each training invocation instead.

Example configurations are provided for `dc`, `dt`, `hgb`, `lr`, `nb`, and `rf`.

The model catalog currently supports:

| Alias | Estimator |
| --- | --- |
| `dc` | `DummyClassifier` |
| `nb` | `GaussianNB` |
| `lr` | `LogisticRegression` |
| `lda` | `LinearDiscriminantAnalysis` |
| `qda` | `QuadraticDiscriminantAnalysis` |
| `knn` | `KNeighborsClassifier` |
| `lsvc` | `LinearSVC` |
| `svc` | RBF `SVC` |
| `dt` | `DecisionTreeClassifier` |
| `ada` | `AdaBoostClassifier` |
| `et` | `ExtraTreesClassifier` |
| `rf` | `RandomForestClassifier` |
| `hgb` | `HistGradientBoostingClassifier` |

Classifier implementations and tuning search spaces are defined in the
[model catalog](src/churn_mlops/models/catalog.py). The feature pipeline adds
custom feature engineering before preprocessing and classification.

Example training configuration:

```yaml
model:
  classifier: dc
  classifier_params:
    random_state: 26

preprocessing:
  numeric_impute_strategy: median
  categorical_impute_strategy: most_frequent

tuning:
  n_trials: 64
  n_splits: 4
  metric: roc_auc
  random_state: 26

calibration:
  n_splits: 3
  n_bins: 10

threshold:
  strategy: max_f1

promotion:
  min_delta: 0.005
  metric_weights:
    roc_auc: 0.2
    pr_auc: 0.2
    f1: 0.2
    brier_score: 0.2
    calibration_error: 0.2
```

Pass the required use case with `--use_case` in the CLI or `use_case=` in a
Python training call. The selected contract supplies the target, time, and
feature columns used by training.

`--n_jobs` sets the total CPU job budget for a run. It defaults to `1`, accepts
positive integers, and uses `-1` to request all available CPUs. During tuning,
trials run in parallel while individual estimators use one job; during
calibration, folds run in parallel while inner estimators use one job. For
uncalibrated fitting, the classifier uses the budget if it supports `n_jobs`.

### Calibration

Calibration is selected per run with `--calibration_method` as `none`,
`sigmoid` (Platt scaling), or `isotonic`; it defaults to `none`. The training
config YAML sets the number of folds and bins used for calibration. For
calibrated runs, `CalibratedClassifierCV` uses expanding historical folds: each
calibration fold is later than the fold's model-fitting data. To compare
calibration approaches, run separate experiments with the same data and
classifier while changing `--calibration_method`.

`LinearSVC` and `SVC` require `--calibration_method` to be `sigmoid` or
`isotonic`.
They do not provide probabilities when used without calibration. During Optuna
ranking-metric evaluation, these estimators are scored using their decision
function.

### Threshold selection

Calibration and Optuna hyperparameter tuning remain restricted to training data.
The final estimator is fitted on the full training split. The threshold
maximizing F1 (`threshold.strategy: max_f1`) is selected from out-of-time test
predictions before test metrics are computed.

Runs log the selected threshold and its OOT selection source as MLflow
parameters. The reliability diagram is logged under `calibration/`.

### Evaluation

After completing hyperparameter tuning (optional), probability calibration and
threshold selection, evaluation metrics like ROC AUC, PR AUC, accuracy,
precision, recall, F1, Brier score, expected calibration error (ECE), fit time,
and prediction time are computed and recorded.

### Registry and promotion

Registration and promotion are selected for each run with `--register-model`
and `--promote-model`; both default to disabled. A run registers an immutable
model version when registration is enabled.

Champion promotion compares the weighted score configured in
`promotion.metric_weights`: ROC AUC, PR AUC, and F1 contribute positively,
while Brier score and ECE are inverted so lower values improve the score.
The model name is `{use_case}-model`, so each use case has its own registered
model and champion lifecycle. The candidate must exceed the champion's score by
`promotion.min_delta`.

Champions from older runs that lack any configured metric are not comparable
and are not automatically replaced; retrain the baseline with calibration
evaluation before enabling automated promotion. The prior champion is retained
under `former_champion`.

Use `--register-model false` or `--no-register-model` to disable registration;
the bare `--register-model` flag enables it. Promotion remains separately
controlled with `--promote-model` / `--no-promote-model`.

### Manifests and model cards

Training writes a manifest linking the MLflow run and its split and evaluation
results. The manifest contains a JSON-safe logged-model reference and identifies
either an exact registered model version or the logged model URI for an
unregistered run. Use the manifest for reproducible batch scoring or serving
instead of relying on a mutable alias. Its versioned JSON groups run and model
identity (including the use case), target/time columns, split metadata, training
details (including calibration method and CPU job budget), evaluation, and
registration/promotion settings. Promotion is marked enabled only when both
registration and promotion are enabled for that run.

Serving validates the manifest's use case against the selected `USE_CASE`;
older manifest schema versions are rejected, so retrain models and regenerate
manifests to use the latest schema.

MLflow runs also include a Markdown model card with model, data, feature,
evaluation, and promotion details, including calibration method, calibration
error, optimal threshold, and threshold-selection strategy. See the
[tracking package](src/churn_mlops/tracking).

### MLflow logged artifacts and run metadata

MLflow run metadata (metrics, parameters, and tags) is separate from files and
models logged to the run. Training runs log evaluation metrics and training
parameters such as classifier, feature, calibration, threshold, registration
and promotion settings.
They also include tags for calibration, threshold-data source, and feature
count. Logged files and models include:

| Training artifact | MLflow location or details |
| --- | --- |
| Fitted scikit-learn pipeline | Logged model named `model`; the manifest records its logged-model URI |
| Training configuration | `config/` |
| Reliability diagram | `calibration/reliability_diagram.png` |
| Training manifest JSON | `manifest/` when manifest writing is enabled (enabled by default) |
| Markdown model card | `documentation/` |

When `monitor` is run with `--mlflow_experiment_name` or `--run_name` (as in the
DVC monitoring pipeline), it logs the summary CSV under `monitoring/` and adds
tags describing its input cohorts and, when available, the model. The command
also writes its report under `OUTPUT_DIR`.

The DVC pipeline for reproducible training experiments on the contrary, writes
separate manifests and predictions under `data/manifests` and `data/predictions`.
The local manifests can be used for scoring and serving. As indicated above,
each manifest is also logged under along with its MLflow run.

To browse run metadata and artifacts, start the MLflow UI using the tracking
database path configured for this project (replace it if `TRACKING_DIR` or
`MLFLOW_TRACKING_URI` is overridden), then open the run's **Artifacts** view:

```bash
uv run mlflow ui --backend-store-uri sqlite:///tracking_local/mlflow.db
```

The UI is available at <http://127.0.0.1:5000> by default. To list files
directly from a run instead, set the tracking URI to the project's database:

```bash
MLFLOW_TRACKING_URI=sqlite:///tracking_local/mlflow.db \
  uv run mlflow artifacts list --run-id RUN_ID
```

The tracking database and artifact files are separate local state. Preserve or
share both the database in `TRACKING_DIR` and files in `ARTIFACT_DIR` to retain
run metadata and artifacts across machines; sharing only one does not provide
a complete MLflow history.

## Runtime configuration

Environment variables can override the runtime defaults:

| Variable | Local default | Purpose |
| --- | --- | --- |
| `USE_CASE` | Required for serving | Registered use_case contract selected by the API |
| `MLFLOW_TRACKING_URI` | SQLite at `tracking_local/mlflow.db` | MLflow tracking and registry |
| `MLFLOW_EXPERIMENT_NAME` | `test` | Default monitoring experiment |
| `MODEL_NAME` | Unset | Registered model name for serving (required unless using a manifest) |
| `MODEL_ALIAS` | `champion` | Alias used to load a serving model |
| `MODEL_VERSION` | Unset | Optional exact model version |
| `MODEL_MANIFEST` | Unset | Optional manifest selecting a logged model |
| `API_HOST` | `127.0.0.1` | Host interface for `serve` |
| `API_PORT` | `8000` | Port for `serve` |
| `REQUEST_ID_HEADER` | `X-Request-ID` | Header used to pass request IDs |
| `LOG_FEATURES` | `True` | Whether validated features are written to API logs |
| `LOG_LEVEL` | Application default | Logging verbosity |

Runtime directories can also be overridden:

| Variable | Local default | Container default |
| --- | --- | --- |
| `DATA_DIR` | `data` | `/app/data` |
| `CONFIG_DIR` | `src/config` | `/app/src/config` |
| `ARTIFACT_DIR` | `artifacts_local` | `/app/artifacts` |
| `TRACKING_DIR` | `tracking_local` | `/app/tracking` |
| `LOGGING_DIR` | `logs_local` | `/app/logs` |
| `OUTPUT_DIR` | `output` | `/app/output` |

By default, the MLflow SQLite database is created inside `TRACKING_DIR`.
`LOG_FEATURES=False` disables feature values in inference logs while retaining
prediction and operational telemetry.

## Run with Docker Compose

The Compose service builds the API image and mounts data, registry state,
artifacts, logs, and outputs from the repository. It mounts `src/config`
read-only. Docker Engine and the Compose plugin are required.

```bash
cp .env.example .env
# Set USE_CASE in .env
docker compose up --build -d
docker compose ps
curl http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
```

The process can start without a registered model: `/health` returns `200`,
while `/ready` and `/predict` return `503` until the configured model can be
loaded. Because the Compose health check uses `/ready`, the container remains
unhealthy until a model is available. After preparing input data and train/test
splits under the mounted `data/` directory, train a model in the same Compose
environment with:

```bash
docker compose run --rm api train --config rf.yaml --split_name oot-26Q2
```

Other CLI commands are available by overriding the default `serve` command,
for example:

```bash
docker compose run --rm api batch-predict \
  --use_case churn \
  --input_dataset partitioned \
  --start_date 2026-01-01 \
  --end_date 2026-06-30
```

Stop the service with `docker compose down`. The mounted host directories
remain in place.

## Development

Run tests and Ruff checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Install the repository's pre-commit hooks with `uv run pre-commit install`.
CI installs locked dependencies, runs Ruff lint and format checks, and runs
the full test suite. Tests generate their own temporary data and isolated
MLflow state.

## License compliance

The project is licensed under MIT; see [LICENSE](LICENSE).

Generate a machine-readable report of installed packages and check it against
[license-policy.json](license-policy.json):

```bash
uv sync --all-groups
uv run pip-licenses --format=json --output-file=license-report.json
uv run python scripts/check_licenses.py license-report.json
```

GPL-licensed packages are reported but do not fail the check. AGPL and SSPL
are rejected, as are proprietary license labels unless the exact package and
reported label have a documented exception. Missing or newly unrecognized
license labels also require review. Exceptions are kept package-specific and
include a reason based on the package's license terms; packages are not
approved solely because they are transitive.

CI runs the same check across all dependency groups and uploads
`license-report.json` as the `dependency-license-report` artifact.
Pull requests also run GitHub Dependency Review against the SPDX
allow-list and package exceptions in
[.github/dependency-review-config.yml](.github/dependency-review-config.yml);
enable the repository's dependency graph for the action to evaluate pull
requests.

## Repository layout

```text
src/churn_mlops/       Application package
  config/              Runtime and training configuration
  data/                Loading, validation, partitioning, and splitting
  evaluation/          Model metrics
  models/              Feature engineering and classifier catalog
  use_cases/           Use-case contracts and domain adapters
  monitoring/          Performance, drift, and operational reports
  serving/             FastAPI service and inference logging
  tracking/            MLflow integration, registry, and manifests
  training/            Training and hyperparameter tuning
src/config/            YAML training configurations
data/raw/              Local input CSVs (not stored in Git)
data/partitioned/      Month-partitioned Parquet data
data/splits/           Train/test Parquet files
monitoring_pipeline/    DVC pipeline for selected monitoring cohorts
notebooks/              Exploratory and analysis notebooks
tests/                  Unit and integration tests
compose.yaml            Local container stack
dvc.yaml                Data preparation, training, and scoring pipeline
params.yaml             Parameters for the root DVC pipeline
```

## Scope and limitations

This repository is a local-first MLOps example. It does not include a shared
MLflow service or artifact store, cloud or Kubernetes deployment configuration,
automated production CD, authentication or authorization, CORS configuration,
or automatically generate data and HTTP schemas from training configuration.
