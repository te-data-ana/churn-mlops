# Churn MLOps

A local-first, end-to-end customer churn prediction project. It demonstrates
data preparation, time-based evaluation, model training and promotion, batch
and HTTP inference, and model, data, and service monitoring.

The project uses scikit-learn pipelines for feature engineering and prediction,
MLflow for experiments and the model registry, DVC for reproducible data and
evaluation pipelines, and FastAPI for serving.

## Quick start

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

Install the project and its development dependencies, then check the CLI:

```bash
uv sync
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
  --input_csv customer_churn_dataset-training.csv \
  --output_csv training.csv \
  --start_date 2025-01-01 \
  --end_date 2025-12-01

uv run python -m churn_mlops.data.manipulation \
  --input_csv customer_churn_dataset-inference.csv \
  --output_csv inference.csv \
  --start_date 2026-01-01 \
  --end_date 2026-06-01
```

`prepare-data` validates a CSV and writes it as a month-partitioned Parquet
dataset under `data/partitioned`. The input argument is a filename under
`data/raw`. Prepare both enriched files:

```bash
uv run churn-mlops prepare-data --input_csv training.csv
uv run churn-mlops prepare-data --input_csv inference.csv
```

Create train and test files from an inclusive date range. Rows before
`split_date` go to training; rows on or after it go to test. By default, the
command reads the `reference_date` column and writes files under `data/splits`.

```bash
uv run churn-mlops create-split \
  --dataset_name partitioned \
  --start_date 2025-04-01 \
  --split_date 2026-04-01 \
  --end_date 2026-06-30 \
  --split_name OOT_26Q2
```

This creates `OOT_26Q2_train.parquet`, `OOT_26Q2_test.parquet`, and split
metadata in `data/splits`.

### Reproduce an out-of-time experiment

The root [dvc.yaml](dvc.yaml) defines a pipeline that adds reference dates,
prepares partitioned data, creates a time split, trains and registers a model,
then scores the held-out cohort. DVC tracks stage inputs and outputs;
MLflow stores runs, metrics, model artifacts, and registered versions.
Optuna tuning, when enabled in the training config, runs as part of training.

The checked-in [params.yaml](params.yaml) configures `OOT_26Q2`: training from
2025-04-01 through 2026-03-31 and testing from 2026-04-01 through 2026-06-30,
using the `rf` classifier. Date bounds are inclusive. Provide the original
training and inference CSVs under `data/raw` before running the pipeline:

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
  --set-param split.name=OOT_26Q2 \
  --set-param 'split.start_date="2025-04-01"' \
  --set-param 'split.split_date="2026-04-01"' \
  --set-param 'split.end_date="2026-06-30"' \
  --set-param model.classifier=lr \
  --set-param model.config=lr.yaml \
  --set-param model.config_path=src/config/lr.yaml \
  --set-param tracking.experiment_name=churn-12m-oot
```

When selecting another classifier, set `model.config` and `model.config_path`
to its YAML file. Each DVC training stage registers a specific model version
without changing the `candidate` or `champion` aliases. Its manifest under
`data/manifests` identifies that version for scoring.

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
  --config rf.yaml \
  --split_name OOT_26Q2 \
  --experiment_name churn-experiments \
  --exclude-columns customerid
```

Configuration filenames resolve under `src/config` by default. The default
split name is `default`. Training configuration controls the target,
engineered features, preprocessing, classifier, evaluation threshold,
optional tuning, and model registration/promotion. See
[Model and tracking reference](#model-and-tracking-reference).

Optuna tuning is disabled by default. To enable it, choose a classifier with a
catalog search space and set tuning options in its YAML configuration:

```yaml
tuning:
  enabled: true
  n_trials: 50
  n_splits: 4
  time_column: reference_date
  metric: roc_auc
  random_state: 26
```

Tuning uses deterministic expanding-window cross-validation on the training
split only. Rows with the same timestamp stay in the same fold. The configured
time column must be present in the training split, and every fold must contain
both target classes. The test split is held out until final evaluation.

### Generate batch predictions

Batch prediction validates input data and loads a registered model. Use a
training manifest or an explicit model name and version when the exact model
version matters; otherwise the configured model alias is used.

For CSV input, the input filename is resolved under `data/raw` and the output
is written under `output`:

```bash
uv run churn-mlops batch-predict \
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
  --input_dataset partitioned \
  --start_date 2026-06-01 \
  --end_date 2026-06-30 \
  --output_dataset batch_predictions
```

The default timestamp column is `reference_date`; the default output dataset
is `batch_predictions`. The result includes the input features and
`predicted_probability`, `predicted_class`, `threshold`, and `model_version`.

To score a specific model version, pass both its registered name and version.
Alternatively, pass a manifest path relative to `DATA_DIR`:

```bash
uv run churn-mlops batch-predict \
  --input_dataset splits/OOT_26Q2_test.parquet \
  --output_dataset predictions/rf_OOT_26Q2 \
  --model_manifest manifests/rf_OOT_26Q2.json
```

The manifest option cannot be combined with `--model_name` or
`--model_version`.

### Serve predictions over HTTP

Start the FastAPI service locally:

```bash
uv run churn-mlops serve --host 127.0.0.1 --port 8000 --reload
```

The service loads the configured registered model once per process. Its
`/health` endpoint is a liveness check; `/ready` reports whether a model loaded
successfully. If model loading fails, the process remains live, but `/ready`
and `/predict` return `503` until it is restarted with a valid model
configuration.

```bash
curl http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
```

Send a customer record to `/predict`:

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

The API uses the `MODEL_NAME` and `MODEL_ALIAS` settings (or an explicit model
version) to load a registered model. The interactive API reference is available
at <http://127.0.0.1:8000/docs>.

For sample-based API checks, the helper reads and samples the inference CSV,
sends each row to the service, and writes predictions with the input features
to the configured output directory:

```bash
uv run python -m churn_mlops.serving.serve_samples \
  --sample_size 1000 \
  --random_state 42 \
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
  --reference_csv output/predictions_training.csv \
  --analysis_csv output/predictions_inference.csv
```

Monitor the configured API prediction and error logs:

```bash
uv run churn-mlops monitor \
  --reference_csv output/predictions_training.csv \
  --api
```

Reports are written under `output` by default. Use `--output_dir` to choose
another location. API logs can be selected explicitly with `--prediction_log`
and `--error_log`.

Monitoring can also use already-scored partitioned datasets under `DATA_DIR`:

```bash
uv run churn-mlops monitor \
  --reference_dataset train_predictions \
  --analysis_dataset api_predictions \
  --error_dataset api_errors \
  --source_reference batch \
  --source_analysis api \
  --output_dir output/monitoring
```

Dataset names are relative to `DATA_DIR`. Supply both reference and analysis
datasets, and set each source to `batch` or `api`; the default analysis source
is `api`. The datasets must already contain prediction columns,
`model_version`, and the `reference_date` (or selected timestamp column).
Reference labels are needed for measured performance; model features are
needed for drift analysis. Error datasets are optional.

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
datasets. By default, these commands read the configured JSONL logs and write
to `data/api_predictions` and `data/api_errors`:

```bash
uv run churn-mlops ingest-prediction-logs
uv run churn-mlops ingest-error-logs
```

Re-ingestion replaces existing partitions for the months represented in the
logs. Use `--dataset_name` or `--timestamp_column` to change the destination
dataset or partition timestamp.

## Data contract

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

Training data also requires a binary `churn` target (`0` or `1`). Depending on
the workflow, data may also contain `customerid` and `reference_date`. CSV
loading normalizes column names; `customerid` can be selected as the CSV index.
Data files are validated with Pandera, and HTTP request features with Pydantic.
See the [data schema](src/churn_mlops/data/schemas.py) and
[API schema](src/churn_mlops/serving/schemas.py).

## Model and tracking reference

### Configuration and classifiers

Training configurations are YAML files under [src/config](src/config). They
define the target, feature-engineering parameters, preprocessing strategies,
classifier parameters, decision threshold, optional tuning, and registry
behavior. Example configurations are provided for `dc`, `dt`, `hgb`, `lr`,
`nb`, and `rf`.

The model catalog currently supports:

| Alias | Estimator |
| --- | --- |
| `dc` | `DummyClassifier` |
| `nb` | `GaussianNB` |
| `lr` | `LogisticRegression` |
| `lda` | `LinearDiscriminantAnalysis` |
| `qda` | `QuadraticDiscriminantAnalysis` |
| `knn` | `KNeighborsClassifier` |
| `lsvc` | Calibrated `LinearSVC` |
| `svc` | Calibrated RBF `SVC` |
| `dt` | `DecisionTreeClassifier` |
| `ada` | `AdaBoostClassifier` |
| `et` | `ExtraTreesClassifier` |
| `rf` | `RandomForestClassifier` |
| `hgb` | `HistGradientBoostingClassifier` |

Classifier implementations and tuning search spaces are defined in the
[model catalog](src/churn_mlops/models/catalog.py). The feature pipeline adds
contract commitment, relative tenure, engagement, yearly spend and support
call rates, late-payer status, and inactive-customer status before
preprocessing and classification.

### Evaluation, registry, and model cards

Training records ROC AUC, PR AUC, accuracy, precision, recall, F1, Brier score,
fit time, and prediction time. The configured threshold determines the
predicted class; ROC AUC is used for model promotion.

When enabled by configuration, a run registers an immutable model version.
Promotion compares candidate ROC AUC with the current champion and changes the
`champion` alias only when the configured improvement threshold is exceeded.
The prior champion is retained under `former_champion`. CLI options can force
registration with `--register-model` or control promotion with
`--promote-model` / `--no-promote-model`.

Training also writes a manifest linking the MLflow run and exact model version
to its split and evaluation results. Use the manifest for reproducible batch
scoring instead of relying on a mutable alias. MLflow runs also include a
Markdown model card with model, data, feature, evaluation, and promotion
details. See the [tracking package](src/churn_mlops/tracking).

## Runtime configuration

Environment variables can override the runtime defaults:

| Variable | Local default | Purpose |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | SQLite at `tracking_local/mlflow.db` | MLflow tracking and registry |
| `MLFLOW_EXPERIMENT_NAME` | `test` | Default training experiment |
| `MODEL_NAME` | `churn-risk` | Registered model name for serving |
| `MODEL_ALIAS` | `champion` | Alias used to load a serving model |
| `MODEL_VERSION` | Unset | Optional exact model version |
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
docker compose run --rm api train --config rf.yaml --split_name OOT_26Q2
```

Other CLI commands are available by overriding the default `serve` command,
for example:

```bash
docker compose run --rm api batch-predict \
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

## Repository layout

```text
src/churn_mlops/       Application package
  config/              Runtime and training configuration
  data/                Loading, validation, partitioning, and splitting
  evaluation/          Model metrics
  models/              Feature engineering and classifier catalog
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
scripts/                Data preparation and serving helpers
tests/                  Unit and integration tests
compose.yaml            Local container stack
dvc.yaml                Data preparation, training, and scoring pipeline
params.yaml             Parameters for the root DVC pipeline
```

## Scope and limitations

This repository is a local-first MLOps example. It does not include a shared
MLflow service or artifact store, cloud or Kubernetes deployment configuration,
automated production CD, authentication or authorization, CORS configuration,
or support for arbitrary input schemas.
