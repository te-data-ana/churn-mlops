#!/usr/bin/env bash

uv run churn-mlops create-split \
  --dataset_name partitioned \
  --timestamp_column reference_date \
  --start_date 2024-01-01 \
  --split_date 2026-01-01 \
  --end_date 2026-06-30 \
  --split_name baseline
