FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MLFLOW_TRACKING_URI=sqlite:////app/tracking/mlflow.db
ENV MLFLOW_EXPERIMENT_NAME=test
ENV ARTIFACT_DIR=/app/artifacts
ENV TRACKING_DIR=/app/tracking
ENV RAW_DATA_DIR=/app/data/raw
ENV CONFIG_DIR=/app/src/config
ENV TMP_DIR=/app/tmp

EXPOSE 8000

ENTRYPOINT ["churn-mlops"]
CMD ["serve"]
