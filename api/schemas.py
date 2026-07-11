from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from enum import Enum

class RiskTier(str, Enum):
    minimal = "minimal"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class TransactionInput(BaseModel):
    # Input schema for POST /predict
    # Raw transaction fields

    transaction_id : str
    user_id : int
    card : int
    timestamp : datetime
    amount : float
    use_chip : str
    merchant_name : int
    merchant_city : str
    merchant_state : str
    mcc : int
    errors : Optional[str] = "None"


class PredictionOutput(BaseModel):
    # Output schema for POST /predict
    # Prediction results, scoring and SHAP (if risk = high / critical)

    transaction_id : str
    fraud_probability : float
    risk_tier : RiskTier
    expected_exposure : float
    is_alert : bool
    inference_latency : float
    shap_explainability : Optional[dict] = None


class AlertRecord(BaseModel):
    # One alert record for GET /alets
    # Joined predictions and transactions tables

    transaction_id : str
    predicted_at : datetime
    model_version : str
    fraud_probability : float
    risk_tier : RiskTier
    expected_exposure : float
    
    user_id : Optional[int] = None
    card : Optional[int] = None
    amount : Optional[float] = None
    merchant_city : Optional[str] = None
    merchant_state : Optional[str] = None
    use_chip : Optional[str] = None
    txn_timestamp : Optional[datetime] = None


class AlertOutput(BaseModel):
    # Output schema for GET /alerts

    alerts : list[AlertRecord]
    total : int


class MetricsOutput(BaseModel):
    # Output schema for GET /metrics

    total_predictions : int
    total_alerts : int
    alert_rate : float
    avg_fraud_prob : float
    avg_exposure : float
    avg_latency_ms : float
    risk_tiers : dict
    alert_over_time : list[dict]