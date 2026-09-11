import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_params: dict | None = None):
        if feature_params is None:
            feature_params = {
                "engagement_window": 30,
                "late_payer_threshold": 15,
                "inactive_threshold": 15,
            }
        self.feature_params = feature_params

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        tenure = X["tenure"].replace(0, np.nan)
        tenure_years = tenure / 12

        contract_map = {
            "Monthly": 1,
            "Quarterly": 4,
            "Biannual": 6,
            "Annual": 12,
        }

        commitment = X["contract_length"].map(contract_map)

        X["contract_commitment"] = commitment.fillna(0)

        X["tenure_rel2_commitment"] = (tenure / commitment).fillna(0)

        X["engagement"] = (
            (X["usage_frequency"] / self.feature_params["engagement_window"])
            * (self.feature_params["engagement_window"] - X["last_interaction"])
        ).fillna(0)

        X["avg_spend_per_year"] = (X["total_spend"] / tenure_years).fillna(0)

        X["avg_s_calls_per_year"] = (X["support_calls"] / tenure_years).fillna(0)

        X["late_payer"] = (
            X["payment_delay"] > self.feature_params["late_payer_threshold"]
        ).astype(int)

        X["inactive_customer"] = (
            X["last_interaction"] > self.feature_params["inactive_threshold"]
        ).astype(int)

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None

        return np.concatenate(
            [
                np.asarray(input_features),
                np.asarray(
                    [
                        "contract_commitment",
                        "tenure_rel2_commitment",
                        "engagement",
                        "avg_spend_per_year",
                        "avg_s_calls_per_year",
                        "late_payer",
                        "inactive_customer",
                    ]
                ),
            ]
        )
