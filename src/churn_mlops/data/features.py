import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        tenure = X["tenure"].replace(0, np.nan)

        contract_map = {
            "Monthly": 1,
            "Quarterly": 4,
            "Annual": 12,
        }

        commitment = X["contract_length"].map(contract_map)

        X["contract_commitment"] = commitment.fillna(0)

        X["tenure_rel2_commitment"] = (tenure / commitment).fillna(0)

        X["engagement"] = (
            X["usage_frequency"] / 30 * (30 - X["last_interaction"])
        ).fillna(0)

        X["avg_spend_per_year"] = (X["total_spend"] / (tenure / 12)).fillna(0)

        X["avg_s_calls_per_year"] = (X["support_calls"] / (tenure / 12)).fillna(0)

        X["late_payer"] = (X["payment_delay"] > 20).astype(int)

        X["inactive_customer"] = (X["last_interaction"] > 15).astype(int)

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
