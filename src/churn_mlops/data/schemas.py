import pandera.pandas as pa
from pandera import Float64, Int64
from pandera.typing import Index, Series


class ChurnDataSchema(pa.DataFrameModel):
    customerid: Index[Int64]
    age: Series[Int64]
    tenure: Series[Int64]
    usage_frequency: Series[Int64]
    support_calls: Series[Int64]
    payment_delay: Series[Int64]
    last_interaction: Series[Int64]
    churn: Series[Int64]
    total_spend: Series[Float64]
    gender: Series[str]
    subscription_type: Series[str]
    contract_length: Series[str]

    class Config:
        coerce = True

    @pa.check("age")
    @classmethod
    def realistic_age(cls, s):
        return s.between(18, 115)

    @pa.check("tenure")
    @classmethod
    def positive_tenure(cls, s):
        return s >= 0

    @pa.check("usage_frequency")
    @classmethod
    def positive_usage_frequency(cls, s):
        return s >= 0

    @pa.check("support_calls")
    @classmethod
    def positive_support_calls(cls, s):
        return s >= 0

    @pa.check("payment_delay")
    @classmethod
    def positive_payment_delay(cls, s):
        return s >= 0

    @pa.check("last_interaction")
    @classmethod
    def positive_last_interaction(cls, s):
        return s >= 0

    @pa.check("total_spend")
    @classmethod
    def positive_total_spend(cls, s):
        return s >= 0

    @pa.check("gender")
    @classmethod
    def valid_gender(cls, s):
        return s.isin(["Female", "Male"])

    @pa.check("subscription_type")
    @classmethod
    def valid_subscription_type(cls, s):
        return s.isin(["Basic", "Standard", "Premium"])

    @pa.check("contract_length")
    @classmethod
    def valid_contract_length(cls, s):
        return s.isin(["Annual", "Monthly", "Quarterly"])

    @pa.check("churn")
    @classmethod
    def valid_target(cls, s):
        return s.isin([0, 1])
