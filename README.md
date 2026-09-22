# Churn MLOps

An end-to-end customer churn prediction project for training, evaluating, registering, and serving tabular machine-learning models.

The project supports three workflows:

- train and evaluate a configurable churn classifier;
- generate batch predictions from a CSV file; and
- serve predictions through a FastAPI application.

Experiments, model artifacts, and promotion decisions are tracked locally with MLflow.

## Quick start

### Requirements

- Python 3.14 or newer
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
	--input_csv customer_churn_dataset-inference.csv \
	--output_csv predictions.csv \
	--index_col customerid
```

Serve the FastAPI application:

```bash
uv run churn-mlops serve --host 127.0.0.1 --port 8000 --reload
```

Training reads configuration files from `src/config` and input data from `data/raw`. Local MLflow state is stored in `tracking/mlflow.db`; run artifacts are written to `artifacts`.

## User workflows

### Train a model

Training loads and validates a CSV, builds a feature and preprocessing pipeline, fits the configured classifier, evaluates it on a stratified holdout set, and logs the result to MLflow.

```bash
uv run churn-mlops train \
	--config sample_training_config.yaml
```

The default training file is `data/raw/customer_churn_dataset-training.csv`.

### Generate batch predictions

Batch inference validates an input CSV, loads the configured registered model, and writes predictions under `tmp`:

```bash
uv run churn-mlops batch-predict \
	--input_csv customer_churn_dataset-inference.csv \
	--output_csv predictions.csv \
	--index_col customerid
```

The output contains:

| Column | Description |
| --- | --- |
| `predicted_probability` | Estimated probability of churn |
| `predicted_class` | Class derived from the configured threshold |
| `threshold` | Threshold used for classification |
| `model_version` | Registered model version used for prediction |

Input filenames are resolved from `data/raw`; output filenames are written to `tmp`.

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

See [tracking implementation](src/churn_mlops/tracking).

## Configuration and environment

Serving defaults can be overridden with environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_NAME` | `churn-propensity` | Registered model name |
| `MODEL_ALIAS` | `champion` | Alias used to load the serving model |
| `LOG_LEVEL` | Application default | Logging verbosity |

Project directories are relative to the repository root:

| Directory | Purpose |
| --- | --- |
| `data/raw` | Input CSV files |
| `src/config` | Training YAML files |
| `tracking` | Local MLflow database |
| `artifacts` | MLflow and model artifacts |
| `tmp` | Batch prediction output |

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
├── data/          # ingestion and schema validation
├── evaluation/    # metrics and timing
├── models/        # features, preprocessing, and classifiers
├── serving/       # FastAPI application and model loading
├── tracking/      # MLflow, registry, and promotion
├── batch_predict.py
├── train.py
└── training.py
src/config/        # YAML training configurations
data/raw/          # input datasets
tests/             # unit and integration tests
tracking/          # local MLflow state
artifacts/         # generated run artifacts
tmp/               # generated batch predictions
```

## Current limitations

This project is designed as a local-first MLOps example. It currently does not provide:

- a remote MLflow tracking server or cloud artifact store;
- Docker, Kubernetes, or cloud deployment configuration;
- automated CI deployment pipeline or production CD configuration;
- authentication, authorization, or CORS configuration;
- production monitoring or drift detection; or
- support for arbitrary input schemas.

Paths are resolved relative to the project root, and the serving API requires
the configured model alias to exist before prediction requests can succeed.
