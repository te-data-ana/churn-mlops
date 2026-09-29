# Churn MLOps

An end-to-end customer churn prediction project for preparing data, training,
evaluating, monitoring, registering, and serving tabular machine-learning
models.

The project supports six workflows:

- prepare and enrich raw datasets for training and inference;
- train and evaluate configurable churn prediction models;
- generate batch predictions from CSV datasets;
- serve predictions through a FastAPI application;
- score sample inference data through the HTTP API; and
- monitor model performance, data drift, and operational metrics.

Experiments, model artifacts, model cards, and promotion decisions are tracked
locally with MLflow.

## Quick start

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

Install dependencies:

```bash
uv sync
```

Run the tests:

```bash
uv run pytest
```

Use the package entry point for the supported local workflows:

```bash
uv run churn-mlops --help
```

Train a model with the sample configuration:

```bash
uv run churn-mlops train \
	--config sample_training_config.yaml
```

Generate batch predictions:

```bash
uv run churn-mlops batch-predict \
	--input_csv inference.csv \
	--output_csv predictions_inference.csv \
	--index_col customerid
```

Serve the FastAPI application:

```bash
uv run churn-mlops serve --host 127.0.0.1 --port 8000 --reload
```

Training reads configuration files from `src/config` and input data from `data/raw`. Local MLflow state is stored in `tracking/mlflow.db`; run artifacts are written to `artifacts`.

## Containerized local stack

Docker Compose provides a reproducible local runtime using the same CLI and
SQLite-backed MLflow registry. The API container binds to `0.0.0.0:8000` and
persists its registry state, model artifacts, and batch outputs on the host.

Requirements:

- Docker Engine with the Compose plugin

Build and start the API:

```bash
cp .env.example .env
docker compose up --build -d
```

The container starts successfully even when no model is available. In that
case `/health` returns `200`, while `/ready` returns `503` until a model has
been trained, registered, and can be loaded from MLflow.

The image entrypoint is the project CLI (`churn-mlops`) and the default command
is `serve`. All workflows exposed by the CLI can be executed inside the
container by overriding the default command, for example:

```bash
docker compose run --rm api train --config sample_training_config.yaml
docker compose run --rm api batch-predict ...
docker compose run --rm api monitor ...
```

Verify the container and endpoints:

```bash
docker compose ps
curl http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
```

Train and register a model in the same Compose environment with:

```bash
docker compose run --rm api train --config sample_training_config.yaml
```

Run batch prediction through the container:

```bash
docker compose run --rm api batch-predict \
	--input_csv inference.csv \
	--output_csv predictions_inference.csv \
	--index_col customerid
```

Monitoring can be run against batch prediction outputs or the API inference
logs produced by the container:

```bash
docker compose run --rm api monitor \
    --reference_csv output/predictions_training.csv \
    --analysis_csv output/predictions_inference.csv
```

```bash
docker compose run --rm api monitor \
    --reference_csv output/predictions_training.csv \
    --api
```

Prediction logs, error logs, batch prediction outputs, and monitoring reports
are persisted on the host through the mounted `logs/` and `output/` directories.

The Compose service mounts `tracking/`, `artifacts/`, and `output/` as writable
directories, and mounts `data/raw` and `src/config` read-only. Keep the MLflow
database and artifacts together: the database contains registry metadata while
the artifacts contain the registered model files. Recreating the API container
does not remove these host directories.

Stop the stack with:

```bash
docker compose down
```

Run the image without Docker Compose:

After building the image, the FastAPI prediction service can be started directly:

```bash
docker run -p 8000:8000 churn-mlops:local
```

## User workflows

### Prepare raw data with reference dates

The project includes a data-preparation helper that validates raw CSV inputs, adds a generated `reference_date` column, and writes the enriched dataset back to disk. This is useful when you want to simulate time-based splits or create training/inference data with a consistent date range.

```bash
uv run python -m churn_mlops.data.manipulation \
	--input_csv customer_churn_dataset-inference.csv \
	--output_csv inference.csv \
	--start_date 2026-01-01 \
	--end_date 2026-08-01
```

The same helper is also used by [scripts/create_data_with_ref_dates.sh](scripts/create_data_with_ref_dates.sh) to generate both training and inference datasets for the repository examples.

### Train a model

Training loads and validates a CSV, builds a feature and preprocessing pipeline, fits the configured classifier, evaluates it on a stratified holdout set, and logs the result to MLflow.

```bash
uv run churn-mlops train \
	--config sample_training_config.yaml
```

The default training file is `data/raw/training.csv`.

### Generate batch predictions

Batch inference validates an input CSV, loads the configured registered model, and writes predictions under `output`:

```bash
uv run churn-mlops batch-predict \
	--input_csv inference.csv \
	--output_csv predictions_inference.csv \
	--index_col customerid
```

The output contains:

| Column | Description |
| --- | --- |
| `predicted_probability` | Estimated probability of churn |
| `predicted_class` | Class derived from the configured threshold |
| `threshold` | Threshold used for classification |
| `model_version` | Registered model version used for prediction |

Input filenames are resolved from `data/raw`; output filenames are written to `output`.

### Monitor predictions

Monitoring compares a labeled reference CSV scored by a model version with either
a scored batch CSV or the live API prediction log. Prefer a held-out or
out-of-time reference cohort over in-sample training predictions.

Monitor batch predictions:

```bash
uv run churn-mlops monitor \
	--reference_csv output/predictions_training.csv \
	--analysis_csv output/predictions_inference.csv
```

Monitor the configured live API prediction and error logs:

```bash
uv run churn-mlops monitor \
	--reference_csv output/predictions_training.csv \
	--api
```

Batch mode writes `monitoring_summary_batch.csv`, API mode uses the corresponding
`_api` suffix.

The monitoring summary is produced at a monthly granularity and separately for
each model version that exists in both the reference and analysis datasets.
The report includes:

- prediction_count
- avg_predicted_probability
- predicted_positive_rate

When ground-truth labels are available, the report additionally contains realized
performance metrics:

- observed_positive_rate
- realized_precision
- realized_recall
- realized_roc_auc

When analysis labels are unavailable and the reference dataset contains labels,
NannyML Confidence-Based Performance Estimation (CBPE) provides estimated
performance metrics:

- cbpe_precision
- cbpe_recall
- cbpe_roc_auc

Feature monitoring includes monthly drift analysis of the model input features
and reports:

- reconstruction_drift_alert_count
- univariate_drift_alert_count

Drift calculations are automatically skipped when the reference and analysis
periods overlap because a proper baseline and monitoring period separation is
required.

For API-based monitoring, operational metrics are also reported:

- api_request_count
- api_error_rate
- prediction_latency_p50_ms
- prediction_latency_p95_ms
- api_error_latency_p50_ms
- api_error_latency_p95_ms

Prediction-error events are incorporated into operational reporting and,
when feature data is available, drift monitoring.

Monitoring is only calculated for model versions present in both reference
and analysis data.

#### Monitoring summary metrics

Depending on the available inputs, the report contains a subset of
the following columns:

| Column | Description |
|----------|-------------|
| period | Indicates whether the record belongs to the reference or analysis dataset |
| model_version | Registered model version being monitored |
| reference_date | Month represented by the monitoring record |
| prediction_count | Number of predictions in the period |
| avg_predicted_probability | Average predicted churn probability |
| predicted_positive_rate | Fraction of predictions classified as churn |
| observed_positive_rate | Actual churn rate (only when labels are available) |
| realized_precision | Measured precision based on observed labels |
| realized_recall | Measured recall based on observed labels |
| realized_roc_auc | Measured ROC AUC based on observed labels |
| cbpe_precision | NannyML estimated precision |
| cbpe_recall | NannyML estimated recall |
| cbpe_roc_auc | NannyML estimated ROC AUC |
| reconstruction_drift_alert_count | Number of reconstruction-drift alerts triggered for the period |
| univariate_drift_alert_count | Number of feature-level drift alerts triggered for the period |
| api_request_count | Total API requests (successful and failed) |
| api_error_rate | Ratio of failed API requests |
| prediction_latency_p50_ms | Median latency of successful prediction requests |
| prediction_latency_p95_ms | 95th percentile latency of successful prediction requests |
| api_error_latency_p50_ms | Median latency of failed prediction requests |
| api_error_latency_p95_ms | 95th percentile latency of failed prediction requests |

Notes:

- Monitoring is performed independently for each model version.
- Monitoring is only performed for model versions that exist in both reference
  and analysis datasets.
- Realized metrics require ground-truth labels.
- CBPE metrics are available only when the reference dataset contains labels.
- Drift metrics are unavailable when reference and analysis periods overlap.
- API-specific metrics are populated only when monitoring API inference logs, otherwise they are omitted.

### Generate sample predictions through the API

For operational checks or smoke tests, the project can sample rows from the inference dataset, send them to the live prediction API, and save the combined response data to a CSV file. This is handled by the sample-serving helper in [src/churn_mlops/serving/serve_samples.py](src/churn_mlops/serving/serve_samples.py) and the convenience script [scripts/serve_samples.sh](scripts/serve_samples.sh).

```bash
uv run python -m churn_mlops.serving.serve_samples --sample_size 5000 --random_state 42
```

The helper loads and validates inference data, requests predictions for each sampled record, and writes a file named like `0042_sample_predictions.csv` (depending on the random state) under the configured output directory.

### Serve predictions through HTTP

Start the FastAPI application through the package CLI:

```bash
uv run churn-mlops serve --host 127.0.0.1 --port 8000 --reload
```

The equivalent direct Uvicorn command is still:

```bash
uv run uvicorn churn_mlops.serving.api:app --reload
```

Check service health:

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"healthy"}
```

The `/health` endpoint is a liveness check. Use `/ready` to verify that the
configured model loaded successfully:

```bash
curl -i http://127.0.0.1:8000/ready
```

The API loads the configured model once during startup and reuses it for
subsequent requests. This happens once per Uvicorn worker process. If the
configured model alias is unavailable, the process stays alive, `/health`
continues to return `200`, and `/ready` and `/predict` return `503` until the
service is restarted with a valid model configuration.

Send one customer record for prediction:

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

The API requires a registered model with the configured alias. By default it loads model `churn-propensity` using the `champion` alias from the MLflow registry.

Access the interactive API docs under http://127.0.0.1:8000/docs, when the FastAPI application is running.

## Input data contract

Training and inference data must contain these customer features:

### Numeric fields

| Field | Constraint |
| --- | --- |
| `age` | Integer from 18 to 115 |
| `tenure` | Non-negative integer |
| `usage_frequency` | Non-negative integer |
| `support_calls` | Non-negative integer |
| `payment_delay` | Non-negative integer |
| `last_interaction` | Non-negative integer |
| `total_spend` | Non-negative number |

### Categorical fields

| Field | Accepted values |
| --- | --- |
| `gender` | `Female`, `Male` |
| `subscription_type` | `Basic`, `Standard`, `Premium` |
| `contract_length` | `Monthly`, `Quarterly`, `Annual` |

Training data must additionally contain a binary `churn` target with values `0` or `1`. CSV data uses `customerid` as its index column when supplied. Pandera validation is applied before training and inference; API requests are validated with Pydantic.

See [data schemas](src/churn_mlops/data/schemas.py) and [API schemas](src/churn_mlops/serving/schemas.py).

## Machine-learning workflow

### Configuration

Training is controlled by YAML files in [src/config](src/config). A configuration contains sections for:

- `data`: target column, test-set size, and random state;
- `feature_builder`: engineered-feature parameters;
- `preprocessing`: numeric and categorical imputation strategies;
- `model`: classifier alias and estimator parameters;
- `evaluation`: prediction threshold; and
- `registry`: model registration and promotion settings.

The repository includes configurations files for `dc`, `dt`, `hgb`, `lr`, `nb`, and `rf`, along with a [sample_training_config.yaml](src/config/sample_training_config.yaml).

### Feature engineering and preprocessing

The pipeline combines domain features with standard preprocessing. Engineered features include contract commitment, relative tenure, engagement, average yearly spend, average yearly support calls, late-payer status, and inactive-customer status.

Numeric values are imputed, categorical values are imputed and one-hot encoded, and scaling is enabled where appropriate for estimator families that benefit from it. Pipeline construction is implemented under [src/churn_mlops/models](src/churn_mlops/models).

### Supported classifiers

| Alias | Estimator |
| --- | --- |
| `dc` | `DummyClassifier` |
| `nb` | `GaussianNB` |
| `lr` | `LogisticRegression` |
| `dt` | `DecisionTreeClassifier` |
| `et` | `ExtraTreesClassifier` |
| `rf` | `RandomForestClassifier` |
| `hgb` | `HistGradientBoostingClassifier` |

The list of supported classifiers can be extended easily, by extending the model catalog [src/churn_mlops/models/catalog.py](src/churn_mlops/models/catalog.py).

### Evaluation

Each run records ROC AUC, PR AUC, accuracy, precision, recall, F1, Brier score, model fit time, and prediction time.

ROC AUC is the metric used for model-promotion decisions. The configured probability threshold controls `predicted_class` in both batch and online inference.

## MLOps capabilities

### Experiment tracking and artifacts

MLflow is configured for local tracking:

- tracking database: `tracking/mlflow.db`;
- artifact directory: `artifacts`; and
- local experiment metadata and run results stored by MLflow.

Runs capture the training configuration, model parameters, evaluation metrics, serialized scikit-learn pipeline, feature names, and pipeline metadata.

### Model registry and promotion

When enabled in the training configuration, the workflow registers a candidate model and evaluates it against the current `champion`.

Promotion requires the candidate ROC AUC to improve on the champion by more than the configured `promotion_delta`. When promotion succeeds, the candidate receives the `champion` alias and the previous champion receives the `former_champion` alias.

The same tracking flow also logs a Markdown model card as an MLflow artifact. The model card summarizes the model metadata, dataset characteristics, feature list, preprocessing choices, evaluation metrics, and the promotion decision when one exists. See [src/churn_mlops/tracking/model_card.py](src/churn_mlops/tracking/model_card.py).

See [tracking implementation](src/churn_mlops/tracking).

### Inference logging and observability

Structured inference logs are collected for every API request to support operational monitoring, debugging, and auditability.

**Prediction events** capture

- request identifier and UTC timestamp,
- prediction latency in milliseconds,
- model name, alias, and registry version,
- predicted probability and predicted class,
- classification threshold used during inference and
- optionally, the validated input features used for prediction.

**Prediction error events** additionally record

- exception type and message and
- full traceback information,

besides request ID, timestamp, latency and (optional) input features.

Example prediction event:

```json
{
	"request_id": "687afb9b-1b20-45ec-8fac-ac53efe4941a",
	"timestamp_utc": "2026-09-23T19:54:19.548767Z",
	"latency_ms": 98.97,
	"features": {"age": 42, "tenure": 18, "usage_frequency": 12, "support_calls": 2, "payment_delay": 0, "last_interaction": 7, "total_spend": 1250.5, "gender": "Female", "subscription_type": "Standard", "contract_length": "Annual"},
	"event": "prediction",
	"model_name": "churn-propensity",
	"model_alias": "champion",
	"model_version": 3,
	"predicted_class": 0,
	"predicted_probability": 0.29,
	"threshold": 0.5
}
```

## Configuration and environment

Runtime and serving defaults can be overridden with environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | Project-local SQLite URI | MLflow tracking and registry backend |
| `MLFLOW_EXPERIMENT_NAME` | `test` | MLflow experiment used for training runs |
| `MODEL_NAME` | `churn-propensity` | Registered model name |
| `MODEL_ALIAS` | `champion` | Alias used to load the serving model |
| `API_HOST` | `127.0.0.1` | Host interface for the `serve` command |
| `API_PORT` | `8000` | Port for the `serve` command |
| `REQUEST_ID_HEADER` | `X-Request-ID` | HTTP header used to propagate request IDs for tracing |
| `LOG_FEATURES` | `True` | Controls whether input features are logged |
| `LOG_LEVEL` | Application default | Logging verbosity |

Runtime directory overrides are also supported. In the container, these
default to the mounted paths shown below:

| Variable | Default in the image | Purpose |
| --- | --- | --- |
| `ARTIFACT_DIR` | `/app/artifacts` | MLflow run artifacts and model files |
| `RAW_DATA_DIR` | `/app/data/raw` | Input CSV files |
| `LOGGING_DIR` | `/app/logs` | JSONL prediction and error event logs |
| `OUTPUT_DIR` | `/app/output` | Batch prediction output |
| `CONFIG_DIR` | `/app/src/config` | Training YAML files |
| `TRACKING_DIR` | `/app/tracking` | SQLite MLflow database |

Without Docker, these settings default to the corresponding directories in the
repository root. `.env.example` contains the container defaults and can be
copied to `.env` for Docker Compose.

An explicit experiment name passed to the training function or CLI takes
precedence over `MLFLOW_EXPERIMENT_NAME`.

### Serving observability configuration

The prediction service supports configurable request tracing and inference-event
logging through environment variables. Every request is assigned a request ID,
either from the configured request header or generated automatically. The
service persists structured prediction and prediction-error events as JSONL
records in the configured logging directory.

When `LOG_FEATURES` is enabled, validated request features are included in
prediction/error logs to support debugging and audit scenarios. For privacy-sensitive
deployments, feature logging can be disabled while retaining latency,
prediction, model-version, and error telemetry.

## Development

Run tests with coverage:

```bash
uv run pytest
```

Run Ruff checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

Install the pre-commit hooks:

```bash
uv run pre-commit install
```

CI runs the locked dependency installation, Ruff lint and format checks, and
the full pytest suite. Integration tests generate temporary input data and
isolated MLflow state, so they do not require committed datasets or local
developer artifacts.

## Repository layout

```text
src/churn_mlops/
├── config/		# configuration helpers
├── data/		# ingestion, validation, and preprocessing
├── evaluation/	# metrics and model evaluation
├── models/		# feature engineering and classifiers
├── monitoring/ # drift, performance, and operational monitoring
├── serving/	# FastAPI application and inference services
├── tracking/	# MLflow integration and model registry
├── training/	# training pipelines and orchestration
├── batch_predict.py
├── train.py
└── __init__.py
src/config/		# YAML training configurations
artifacts/		# generated run artifacts
data/raw/		# input datasets
logs/			# API prediction and error logs
notebooks/		# exploratory notebooks
output/			# generated sample and batch prediction outputs
scripts/		# helper scripts for data prep and sample-serving workflows
tests/			# unit and integration tests
tracking/		# local MLflow state
```

## Current limitations

This project is designed as a local-first MLOps example. It currently does not provide:

- a remote MLflow tracking server or cloud artifact store;
- Kubernetes, or cloud deployment configuration;
- automated production CD configuration;
- authentication, authorization, or CORS configuration;
- support for arbitrary input schemas.

Paths are resolved relative to the project root, and the serving API requires
the configured model alias to exist before prediction requests can succeed.
