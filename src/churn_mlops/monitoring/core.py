"""Load prediction data, calculate monitoring metrics, and write reports."""

import logging
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any

import nannyml as nml
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from churn_mlops.config import RuntimeSettings, ServingSettings, configure_logging
from churn_mlops.data import normalize_strings, read_jsonl_prediction_log, validate_data
from churn_mlops.serving.schemas import InputFeatures

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = list(InputFeatures.model_fields)
PRED_PROBA_COLUMN = "predicted_probability"
PRED_CLASS_COLUMN = "predicted_class"
PREDICTION_COLUMNS = [PRED_PROBA_COLUMN, PRED_CLASS_COLUMN]
TARGET_COLUMN = "churn"
PERIOD_COLUMN = "period"
MODEL_VERSION_COLUMN = "model_version"
TIMESTAMP_COLUMN = "reference_date"
AGGREGATION_COLUMNS = [PERIOD_COLUMN, MODEL_VERSION_COLUMN, TIMESTAMP_COLUMN]


@dataclass(frozen=True)
class MonitoringReport:
    """Monitoring summary and detailed NannyML results."""

    summary: pd.DataFrame
    summary_path: Path | None = None


def _read_scored_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Monitoring CSV does not exist: {path}")

    frame = pd.read_csv(path)
    frame.columns = normalize_strings(frame.columns)
    return frame


def _event_log_to_frame(path: Path, event: str) -> pd.DataFrame:
    if event == "prediction" and not path.exists():
        raise ValueError(f"{event} log path '{path}' does not exist.")

    frame = read_jsonl_prediction_log(jsonl_path=path)

    if frame.empty:
        logger.warning(f"{event} log '{path}' does not contain any (valid) records.")
        return frame
    else:
        all_records_prediction_events = (frame["event"] == event).all()
        if not all_records_prediction_events:
            raise ValueError(f"Log '{path}' contains events other than {event}.")

    required = {"timestamp_utc", MODEL_VERSION_COLUMN, "latency_ms"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{event} event is missing columns: {', '.join(sorted(missing))}."
        )

    frame.rename(columns={"timestamp_utc": TIMESTAMP_COLUMN}, inplace=True)

    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN], utc=True, errors="raise"
    ).dt.tz_localize(None)
    frame["latency_ms"] = pd.to_numeric(frame["latency_ms"], errors="coerce")
    return frame


def _prepare_validated_frame(
    frame: pd.DataFrame,
    name: str,
    source: str,
    require_predictions: bool = True,
    require_target: bool = False,
    require_features: bool = False,
) -> pd.DataFrame:
    frame.columns = normalize_strings(frame.columns)

    frame.insert(0, "source", source)
    frame.insert(1, PERIOD_COLUMN, name)

    required = set(AGGREGATION_COLUMNS)
    if require_predictions:
        required.union(*PREDICTION_COLUMNS)
    if require_target:
        required.add(TARGET_COLUMN)
    missing = required.difference(set(frame.columns))
    if missing:
        raise ValueError(
            f"{name} data is missing required columns: {', '.join(sorted(missing))}."
        )

    if require_predictions:
        frame[PRED_PROBA_COLUMN] = pd.to_numeric(
            frame[PRED_PROBA_COLUMN], errors="raise"
        )
        if not frame[PRED_PROBA_COLUMN].between(0, 1).all():
            raise ValueError(f"{name} data contains probabilities outside [0, 1].")
        frame[PRED_CLASS_COLUMN] = pd.to_numeric(
            frame[PRED_CLASS_COLUMN], errors="raise"
        ).astype(int)
        if not frame[PRED_CLASS_COLUMN].isin([0, 1]).all():
            raise ValueError(
                f"{name} data contains predicted classes outside {{0, 1}}."
            )

    if require_target and frame[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"{name} data must contain a target label for every row / record."
        )
    if require_target and frame[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            f"{name} data must contain both target classes for binary classification."
        )

    if require_features:
        missing_features = set(FEATURE_COLUMNS).difference(set(frame.columns))
        if missing_features:
            raise ValueError(
                f"{name} data is missing model input / feature columns: "
                f"{', '.join(sorted(missing_features))}."
            )
        _ = validate_data(frame)

    frame[TIMESTAMP_COLUMN] = (
        pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="raise")
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    frame[MODEL_VERSION_COLUMN] = frame[MODEL_VERSION_COLUMN].astype(int)

    return frame


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    flattened = frame.copy()
    if isinstance(flattened.columns, pd.MultiIndex):
        flattened.columns = [
            "__".join(str(part) for part in column if str(part))
            for column in flattened.columns
        ]
    else:
        flattened.columns = [str(column) for column in flattened.columns]
    return flattened.reset_index(drop=False)


def _extract_columns(frame: pd.DataFrame, name_set: set) -> list:
    return [column for column in frame.columns if column.split("__")[-1] in name_set]


# def _result_details(
#     result: Any, model_version: int, calculator_name: str
# ) -> pd.DataFrame:
#     detail = _flatten_columns(result.to_df())
#     period_column = _extract_columns(frame=detail, name_set={"period"})[0]
#     date_column = _extract_columns(frame=detail, name_set={"start_date", "key"})[0]
#     if period_column:
#         detail[PERIOD_COLUMN] = detail[period_column]
#     detail[MODEL_VERSION_COLUMN] = model_version
#     if date_column:
#         detail[TIMESTAMP_COLUMN] = pd.to_datetime(detail[date_column], errors="coerce")
#     detail["calculator"] = calculator_name
#     return detail


def _drift_alert_summary(
    result: Any, model_version: int, calculator_name: str
) -> pd.DataFrame:
    raw = _flatten_columns(result.to_df())

    period_columns = _extract_columns(frame=raw, name_set={"period"})
    date_columns = _extract_columns(frame=raw, name_set={"start_date", "key"})
    alert_columns = _extract_columns(frame=raw, name_set={"alert"})

    if not period_columns:
        raise KeyError(f"'period' not in {calculator_name} results.")
    if not date_columns:
        raise KeyError(
            f"Date column could not be identified from {calculator_name} results."
        )
    if not alert_columns:
        raise KeyError(f"No 'alert' columns in {calculator_name} results.")

    dates = pd.to_datetime(raw[date_columns[0]], errors="coerce")
    alerts = raw[alert_columns].fillna(False).astype(bool).sum(axis=1)

    return pd.DataFrame(
        {
            PERIOD_COLUMN: raw[period_columns[0]],
            MODEL_VERSION_COLUMN: model_version,
            TIMESTAMP_COLUMN: dates,
            f"{calculator_name}_alert_count": alerts,
        }
    )


def _cbpe_summary(
    result: Any, model_version: int, calculator_name: str
) -> pd.DataFrame:
    raw = _flatten_columns(result.to_df())

    period_columns = _extract_columns(frame=raw, name_set={"period"})
    date_columns = _extract_columns(frame=raw, name_set={"start_date", "key"})
    value_columns = _extract_columns(frame=raw, name_set={"value"})

    if not period_columns:
        raise KeyError(f"'period' not in {calculator_name} results.")
    if not date_columns:
        raise KeyError(
            f"Date column could not be identified from {calculator_name} results."
        )
    if not value_columns:
        raise KeyError(f"No estimated metric values in {calculator_name} results.")

    dates = pd.to_datetime(raw[date_columns[0]], errors="coerce")

    summary = pd.DataFrame(
        {
            PERIOD_COLUMN: raw[period_columns[0]],
            MODEL_VERSION_COLUMN: model_version,
            TIMESTAMP_COLUMN: dates,
        }
    )
    for col in value_columns:
        col_name = calculator_name + "_" + col.split("__")[0]
        summary[col_name] = raw[col]
    return summary


def _base_summary(
    reference: pd.DataFrame,
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat([reference, analysis], ignore_index=True)
    grouped = combined.groupby(["source", *AGGREGATION_COLUMNS], dropna=False)
    summary = grouped.agg(
        prediction_count=(PRED_CLASS_COLUMN, "size"),
        avg_predicted_probability=(PRED_PROBA_COLUMN, "mean"),
        predicted_positive_rate=(PRED_CLASS_COLUMN, "mean"),
    ).reset_index()
    return summary


def _actual_metrics(
    reference: pd.DataFrame,
    analysis: pd.DataFrame,
) -> pd.DataFrame:

    frame = pd.concat([reference, analysis], ignore_index=True)

    if TARGET_COLUMN not in frame.columns:
        return pd.DataFrame(columns=AGGREGATION_COLUMNS)

    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(AGGREGATION_COLUMNS, dropna=False)
    for (period, model_version, month), chunk in grouped:
        chunk = chunk.dropna(subset=[TARGET_COLUMN])
        if chunk.empty:
            continue
        target = chunk[TARGET_COLUMN].astype(int)
        predicted = chunk[PRED_CLASS_COLUMN].astype(int)
        probability = chunk[PRED_PROBA_COLUMN].astype(float)
        rows.append(
            {
                PERIOD_COLUMN: period,
                MODEL_VERSION_COLUMN: model_version,
                TIMESTAMP_COLUMN: month,
                "observed_positive_rate": float(target.mean()),
                "realized_precision": float(
                    precision_score(target, predicted, zero_division=0)
                ),
                "realized_recall": float(
                    recall_score(target, predicted, zero_division=0)
                ),
                "realized_roc_auc": (
                    float(roc_auc_score(target, probability))
                    if target.nunique() == 2
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _calculate_version_drift(
    reference: pd.DataFrame,
    analysis: pd.DataFrame,
    errors: pd.DataFrame,
    model_version: int,
) -> pd.DataFrame:

    monitoring = pd.concat([analysis, errors], ignore_index=True)

    continuous_features = (
        monitoring[FEATURE_COLUMNS].select_dtypes(include=["number"]).columns.tolist()
    )

    categorical_features = (
        monitoring[FEATURE_COLUMNS]
        .select_dtypes(include=["category", "object", "str", "string"])
        .columns.tolist()
    )

    alerts = []

    if TARGET_COLUMN in reference.columns:
        cbpe = (
            nml.CBPE(
                problem_type="classification_binary",
                y_pred_proba=PRED_PROBA_COLUMN,
                y_pred=PRED_CLASS_COLUMN,
                y_true=TARGET_COLUMN,
                timestamp_column_name=TIMESTAMP_COLUMN,
                metrics=["precision", "recall", "roc_auc"],
                chunk_period="M",
            )
            .fit(reference)
            .estimate(analysis)
        )
        alerts.append(_cbpe_summary(cbpe, model_version, "cbpe"))

    reconstruction = (
        nml.DataReconstructionDriftCalculator(
            column_names=FEATURE_COLUMNS,
            timestamp_column_name=TIMESTAMP_COLUMN,
            chunk_period="M",
        )
        .fit(reference)
        .calculate(monitoring)
    )
    alerts.append(
        _drift_alert_summary(reconstruction, model_version, "reconstruction_drift")
    )

    univariate = (
        nml.UnivariateDriftCalculator(
            column_names=FEATURE_COLUMNS,
            treat_as_numerical=continuous_features,
            treat_as_categorical=categorical_features,
            timestamp_column_name=TIMESTAMP_COLUMN,
            chunk_period="M",
            continuous_methods=["jensen_shannon"],
            categorical_methods=["jensen_shannon"],
        )
        .fit(reference)
        .calculate(monitoring)
    )
    alerts.append(_drift_alert_summary(univariate, model_version, "univariate_drift"))

    alerts_combined = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on=AGGREGATION_COLUMNS,
            how="outer",
        ),
        alerts,
    )

    return alerts_combined


def _operational_summary(
    analysis: pd.DataFrame,
    errors: pd.DataFrame,
) -> pd.DataFrame:

    analysis_grouped = analysis.groupby(AGGREGATION_COLUMNS, dropna=False)
    success_summary = analysis_grouped.agg(
        api_success_count=("latency_ms", "size"),
        prediction_latency_p50_ms=("latency_ms", "median"),
        prediction_latency_p95_ms=("latency_ms", lambda values: values.quantile(0.95)),
    ).reset_index()

    errors_grouped = errors.groupby(AGGREGATION_COLUMNS, dropna=False)
    error_summary = errors_grouped.agg(
        api_error_count=("latency_ms", "size"),
        api_error_latency_p50_ms=("latency_ms", "median"),
        api_error_latency_p95_ms=("latency_ms", lambda values: values.quantile(0.95)),
    ).reset_index()

    api_totals = success_summary.merge(
        error_summary, on=AGGREGATION_COLUMNS, how="outer"
    )
    api_totals["api_success_count"] = (
        api_totals["api_success_count"].fillna(0).astype(int)
    )
    api_totals["api_error_count"] = api_totals["api_error_count"].fillna(0).astype(int)
    api_totals["api_request_count"] = (
        api_totals["api_success_count"] + api_totals["api_error_count"]
    )
    api_totals["api_error_rate"] = api_totals["api_error_count"] / api_totals[
        "api_request_count"
    ].replace(0, pd.NA)
    return api_totals.drop(columns=["api_success_count", "api_error_count"])


def build_monitoring_report(
    reference: pd.DataFrame,
    analysis: pd.DataFrame,
    *,
    source_reference: str = "batch",
    source_analysis: str = "api",
    drift: bool = True,
    errors: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> MonitoringReport:
    """Calculate model monitoring metrics based on normalized data frames.

    Args:
        reference: Labeled reference cohort containing prediction results from
            the model version(s) to be reviewed.
        analysis: Scored analysis cohort containing model's prediction results
            for inference time period.
        source_reference: Either ``batch`` or ``api``.
        source_analysis: Either ``batch`` or ``api``.
        drift: Boolean flag to indicate if drift KPIs should be calculated.
        errors: Optional API prediction error-event data frame.
        output_dir: Optional directory receiving CSV reports.

    Returns:
        Model monitoring report, for time period covered by ``analysis`` dataframe.
        Monitoring metrics are summarized by month and cover different domains like
        prediction, observed, operational and drift KPIs, depending on ``source``
        and provided input data frames.

    Raises:
        ValueError: If required data fields are missing, ``source`` is unsupported or
            ``reference`` and ``analysis`` frames do not have common model version(s).
    """

    if source_reference not in {"batch", "api"}:
        raise ValueError("``source_reference`` must be either 'batch' or 'api'.")
    if source_analysis not in {"batch", "api"}:
        raise ValueError("``source_analysis`` must be either 'batch' or 'api'.")

    reference_frame = _prepare_validated_frame(
        frame=reference.copy(),
        name="reference",
        source=source_reference,
        require_predictions=True,
        require_target=True,
        require_features=drift,
    )
    analysis_frame = _prepare_validated_frame(
        frame=analysis.copy(),
        name="analysis",
        source=source_analysis,
        require_predictions=True,
        require_target=False,
        require_features=drift,
    )
    if errors is None or errors.empty:
        errors_init = pd.DataFrame(columns=analysis_frame.columns).drop(
            columns=[PERIOD_COLUMN, "source"]
        )
    else:
        errors_init = errors.copy()
    errors_frame = _prepare_validated_frame(
        frame=errors_init,
        name="analysis",
        source=source_analysis,
        require_predictions=False,
        require_target=False,
        require_features=drift,
    )

    # Model monitoring reports can be calculated for multiple model versions simultaneously
    # For that purpose, model versions need to be present in both reference and analysis frames
    unique_versions_reference = set(reference_frame[MODEL_VERSION_COLUMN].unique())
    unique_versions_analysis = set(analysis_frame[MODEL_VERSION_COLUMN].unique())

    versions_common = sorted(
        unique_versions_reference.intersection(unique_versions_analysis)
    )
    versions_only_reference = sorted(
        unique_versions_reference.difference(unique_versions_analysis)
    )
    versions_only_analysis = sorted(
        unique_versions_analysis.difference(unique_versions_reference)
    )

    if not versions_common:
        raise ValueError(
            "Reference and analysis frames do not cover common model versions."
        )
    if versions_common:
        logger.info(
            "Report can be built for model versions %s,"
            "which are present in both, reference and analysis data.",
            versions_common,
        )
    if versions_only_reference:
        for model_version in versions_only_reference:
            logger.warning("No analysis cohort for model version %s.", model_version)
    if versions_only_analysis:
        for model_version in versions_only_analysis:
            logger.warning("No reference cohort for model version %s.", model_version)

    reference_frame = reference_frame.loc[
        reference_frame[MODEL_VERSION_COLUMN].isin(versions_common), :
    ]
    analysis_frame = analysis_frame.loc[
        analysis_frame[MODEL_VERSION_COLUMN].isin(versions_common), :
    ]

    # reference and analysis periods should not overlapp
    ref_max = reference_frame[TIMESTAMP_COLUMN].max()
    ana_min = min(
        analysis_frame[TIMESTAMP_COLUMN].min(), errors_frame[TIMESTAMP_COLUMN].min()
    )
    if ana_min <= ref_max:
        logger.warning(
            f"Reference and analysis periods overlap: {ref_max=} >= {ana_min=}"
        )
        logger.warning("Due to overlapp, drift analysis will not be executed.")
        drift = False

    # To be able to compare monitoring KPIs across analysis and reference period,
    # they are calculated for both, if applicable
    summary = _base_summary(reference=reference_frame, analysis=analysis_frame)

    actual = _actual_metrics(reference=reference_frame, analysis=analysis_frame)
    if not actual.empty:
        summary = summary.merge(actual, on=AGGREGATION_COLUMNS, how="left")

    # Drift analysis separated strictly by model verison and executed only
    # if reference and analysis time periods have no overlapp.
    # Data from failed prediction attempts should also be considered.
    alert_frames: list[pd.DataFrame] = []
    if drift:
        for model_version in versions_common:
            version_reference = reference_frame.loc[
                reference_frame[MODEL_VERSION_COLUMN] == model_version
            ]
            version_analysis = analysis_frame.loc[
                analysis_frame[MODEL_VERSION_COLUMN] == model_version
            ]
            version_errors = errors_frame.loc[
                errors_frame[MODEL_VERSION_COLUMN] == model_version
            ]
            version_alerts = _calculate_version_drift(
                reference=version_reference,
                analysis=version_analysis,
                errors=version_errors,
                model_version=model_version,
            )
            alert_frames.append(version_alerts)

    # Alerts for different model versions need to be combined and merged
    if alert_frames:
        alerts = pd.concat(alert_frames, ignore_index=True, sort=False)
        summary = summary.merge(alerts, on=AGGREGATION_COLUMNS, how="left")

    # Operational KPIs are only available for API based predictions
    if source_analysis == "api":
        operational = _operational_summary(analysis=analysis_frame, errors=errors_frame)
        summary = summary.merge(operational, on=AGGREGATION_COLUMNS, how="outer")

    summary = summary.sort_values([MODEL_VERSION_COLUMN, TIMESTAMP_COLUMN]).reset_index(
        drop=True
    )

    summary_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / f"monitoring_summary_{source_analysis}.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Monitoring summary written to '%s'.", summary_path)

    return MonitoringReport(summary, summary_path)


def run_monitoring(
    reference_csv: Path,
    *,
    analysis_csv: Path | None = None,
    prediction_log: Path | None = None,
    error_log: Path | None = None,
    output_dir: Path | None = None,
) -> MonitoringReport:
    """Load monitoring inputs and write a report for batch or API predictions.

    Args:
        reference_csv: Path to labeled CSV file containing prediction results from
            the model version(s) to be reviewed, based on data covering model
            training period or out-of-time sample.
        analysis_csv: Optional path to CSV file containing model's prediction
            results for inference time period (after training data period).
            Mutually exclusive with ``prediction_log``.
        prediction_log: Optional path to API prediction logs in JSONL format.
            Mutually exclusive with ``analysis_csv``.
        error_log: Optional path to API prediction error logs in JSONL format.
            Only used in combination with ``prediction_log``.
        output_dir: Output directory. Defaults to ``RuntimeSettings.output_dir``.

    Returns:
        Model monitoring report, for time period covered by ``analysis_csv`` /
        ``prediction_log``. Monitoring metrics are summarized by month and cover
        different domains like prediction, observed, operational and drift KPIs,
        depending on provided input data.

    Raises:
        ValueError: If the analysis source is missing or ambiguous.
    """
    if (analysis_csv is None) == (prediction_log is None):
        raise ValueError("Provide exactly one of analysis_csv or prediction_log.")
    if error_log is not None and prediction_log is None:
        raise ValueError("error_log can only be used with prediction_log.")

    settings = RuntimeSettings()
    report_dir = output_dir or settings.output_dir
    reference = _read_scored_csv(reference_csv)
    if analysis_csv is not None:
        return build_monitoring_report(
            reference=reference,
            analysis=_read_scored_csv(analysis_csv),
            source_reference="batch",
            source_analysis="batch",
            output_dir=report_dir,
        )
    else:
        return build_monitoring_report(
            reference=reference,
            analysis=_event_log_to_frame(path=prediction_log, event="prediction"),
            source_reference="batch",
            source_analysis="api",
            errors=_event_log_to_frame(path=error_log, event="prediction_error"),
            output_dir=report_dir,
        )


def main() -> None:
    """Parse CLI arguments and run the requested monitoring workflow."""
    import argparse

    settings = ServingSettings()
    parser = argparse.ArgumentParser(description="Monitor batch or API predictions.")
    parser.add_argument("--reference_csv", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--analysis_csv", type=Path)
    source.add_argument("--prediction_log", type=Path)
    source.add_argument("--api", action="store_true")
    parser.add_argument("--error_log", type=Path)
    parser.add_argument("--output_dir", type=Path, default=settings.output_dir)
    args = parser.parse_args()

    prediction_log = None
    error_log = None
    if args.api or args.prediction_log is not None:
        prediction_log = args.prediction_log or settings.prediction_log_path
        error_log = args.error_log or settings.error_log_path
    run_monitoring(
        reference_csv=args.reference_csv,
        analysis_csv=args.analysis_csv,
        prediction_log=prediction_log,
        error_log=error_log,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    configure_logging()
    main()
