#!/usr/bin/env bash

uv run python -m churn_mlops.data.manipulation \
    --input_csv customer_churn_dataset-inference.csv \
    --output_csv inference.csv \
    --start_date 2026-01-01 \
    --end_date 2026-08-01

uv run python -m churn_mlops.data.manipulation \
    --input_csv customer_churn_dataset-training.csv \
    --output_csv training.csv \
    --start_date 2024-01-01 \
    --end_date 2025-12-01
