#!/usr/bin/env bash

uv run churn-mlops prepare-data \
    --input_csv training.csv \
    --timestamp_column reference_date

uv run churn-mlops prepare-data \
    --input_csv inference.csv \
    --timestamp_column reference_date
