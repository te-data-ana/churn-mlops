#!/usr/bin/env bash

uv run uvicorn churn_mlops.serving.api:app & API_PID=$!

until curl -s http://127.0.0.1:8000/health > /dev/null; do
    sleep 1
done

uv run python -m churn_mlops.serving.serve_samples --sample_size 5000

kill $API_PID
