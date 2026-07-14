import pandas as pd
import numpy as np
import logging
import os
from dotenv import load_dotenv
import time
from sqlalchemy import text
import yaml
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from api.schemas import TransactionInput, PredictionOutput, RiskTier

from slowapi import Limiter
from slowapi.util import get_remote_address

from src.database.db import engine, insert_prediction, insert_scored_transaction
from src.risk.scoring import compute_risk_tier, compute_expected_exposure
from src.explainability.shap_explainer import explain_transaction

from src.features.feature_utils import (
    compute_temp_features,
    compute_time_since_last_txn,
    apply_mcc_encoding,
    compute_velocity_features,
    compute_spend_features,
    compute_zscore,
    compute_is_new_merchant,
    compute_cross_card_features,
    compute_is_new_state,
    compute_is_new_city,
    apply_cold_start_values,
    compute_raw_features
)

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter()

MODEL_VERSION = os.getenv("MODEL_VERSION")

limiter = Limiter(key_func=get_remote_address)

# Loading path and config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

RATE_LIMIT = f"{_config['api']['rate_limit_predict']}/minute"


def is_localhost(request: Request) -> bool:
    host = request.client.host

    return (
        host in ("127.0.0.1", "::1", "localhost")
        or host.startswith("172.")
        or host.startswith("192.168.")
        or host.startswith("10.")
    )


def fetch_user_card_history(
    user_id: int, 
    card: int, 
    before_timestamp: pd.Timestamp
) -> pd.DataFrame:
    # Fetching past 365 days history for specific user-card pair

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT
                 user_id, card, timestamp, amount, is_refund,
                 merchant_name, merchant_state, merchant_city, mcc,
                 use_chip, errors
                FROM transactions
                WHERE user_id = :user_id
                 AND card = :card
                 AND timestamp < :ts
                 AND timestamp >= :ts_365d
                ORDER BY timestamp ASC
            """),
            {
                "user_id" : user_id,
                "card" : card,
                "ts" : before_timestamp,
                "ts_365d" : before_timestamp - pd.Timedelta(days=365)
            }
        ).fetchall()


    if not rows:
        return pd.DataFrame(columns=[
            "user_id", "card", "timestamp", "amount", "is_refund",
            "merchant_name", "merchant_state", "merchant_city",
            "mcc", "use_chip", "errors"
        ])
    return pd.DataFrame([dict(r._mapping) for r in rows])


def fetch_cross_card_history(user_id: int, before_timestamp: pd.Timestamp) -> pd.DataFrame:
    # Fetching past 24 hour history for cross-card features

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT user_id, card, timestamp
                FROM transactions
                WHERE user_id = :user_id
                 AND timestamp < :ts
                 AND timestamp >= :ts_1440min
                ORDER BY timestamp ASC
            """),
            {
                "user_id" : user_id,
                "ts" : before_timestamp,
                "ts_1440min" : before_timestamp - pd.Timedelta(minutes=1440)
            }
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["user_id", "card", "timestamp"])
    return pd.DataFrame([dict(r._mapping) for r in rows])


def build_features(transaction: TransactionInput, request: Request) -> pd.DataFrame:
    # Building 1 row df for the incoming txn

    ts = pd.Timestamp(transaction.timestamp)
    history = fetch_user_card_history(transaction.user_id, transaction.card, ts)
    cross_card_history = fetch_cross_card_history(transaction.user_id, ts)

    cfg = request.app.state.config
    VELOCITY_WINDOWS_MIN = cfg["features"]["velocity_windows_min"]
    BASELINE_WINDOW_DAYS = cfg["features"]["baseline_window_days"]
    CROSS_CARDS_MIN = cfg["features"]["cross_cards_min"]


    # Building df (1 row)
    raw = compute_raw_features(
        amount= transaction.amount,
        errors= transaction.errors,
        high_value_threshold= cfg["features"]["high_value_threshold"]
    )

    current = pd.DataFrame([{
        "user_id" : transaction.user_id,
        "card" : transaction.card,
        "timestamp" : ts,
        "amount" : transaction.amount,
        "merchant_name" : transaction.merchant_name,
        "merchant_state" : transaction.merchant_state,
        "merchant_city" : transaction.merchant_city,
        "mcc" : transaction.mcc,
        "use_chip" : transaction.use_chip,
        "errors" : transaction.errors or "None",
        **raw
    }])


    # Concatenating current txn and txn history
    df = (
        pd.concat([history, current])
        .sort_values(["user_id", "card", "timestamp"])
        .reset_index(drop=True)
    )

    df["is_refund"] = df["is_refund"].fillna(0).astype(bool).astype("int8")


    # Computing Features
    # Temporal Features
    df = compute_temp_features(df)
    df["minutes_since_last_txn"] = compute_time_since_last_txn(df)

    # MCC Encoding (training data mcc)
    df["mcc_frequency"] = apply_mcc_encoding(df, request.app.state.mcc_freq)


    # Card Features
    for window in VELOCITY_WINDOWS_MIN:
        col_name = f"card_txn_count_{window}min"
        
        df[col_name] = compute_velocity_features(
            df[['timestamp', "is_refund"]], 
            window_min=window
            ).astype("float32")


    for window in BASELINE_WINDOW_DAYS:
        col_name = f"card_txn_count_{window}d"
        df[col_name] = compute_velocity_features(
            df[['timestamp', "is_refund"]], 
            window_min=window * 1440
            ).astype("float32")

    for window in BASELINE_WINDOW_DAYS:
        card_spend_features = (
            compute_spend_features(
                df[["timestamp", "amount", "is_refund"]], 
                'card', 
                window
            ).astype("float32")
        )
        df[card_spend_features.columns] = card_spend_features


    # User Features
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    for window in VELOCITY_WINDOWS_MIN:
        col_name = f"user_txn_count_{window}min"
        
        df[col_name] = compute_velocity_features(
            df[['timestamp', "is_refund"]], 
            window_min=window
            ).astype("float32")

    for window in BASELINE_WINDOW_DAYS:
        col_name = f"user_txn_count_{window}d"
        
        df[col_name] = compute_velocity_features(
            df[['timestamp', "is_refund"]], 
            window_min=window * 1440
            ).astype("float32")

    for window in BASELINE_WINDOW_DAYS:
        user_spend_features = (
            compute_spend_features(
                df[["timestamp", "amount", "is_refund"]], 
                'user', 
                window
            ).astype("float32")
        )
        df[user_spend_features.columns] = user_spend_features


    # Z-Score Features
    for window in BASELINE_WINDOW_DAYS:
        df[f"card_amount_zscore_{window}d"] = compute_zscore(df, "card", window)        
        df[f"user_amount_zscore_{window}d"] = compute_zscore(df, "user", window)

    # Dropping weak features
    drop_cols = [
        "card_spend_mean_365d", 
        "card_spend_std_365d", 
        "user_spend_mean_365d", 
        "user_spend_std_365d"
    ]
    df = df.drop(columns=drop_cols)


    # Merchant Familiarity Features
    merchant_df = df[["user_id", "card", "merchant_name"]].copy()
    df["card_is_new_merchant"] = compute_is_new_merchant(merchant_df, 'card', ['user_id', 'card'])
    df["user_is_new_merchant"] = compute_is_new_merchant(merchant_df, 'user', ['user_id'])
    
    # Cross-Card Level Features
    cross_card_current = current[["user_id", "card", "timestamp"]].copy()
    
    if cross_card_history.empty:
        cross_card_df = cross_card_current.copy()
    else:
        cross_card_df = (
            pd.concat([cross_card_history, cross_card_current])
            .sort_values(["user_id", "timestamp"])
            .reset_index(drop=True)
        )
    
    cross_card_df = cross_card_df.groupby("user_id", sort=False)
    cross_card_series = compute_cross_card_features(cross_card_df, CROSS_CARDS_MIN)
    df[f"distinct_cards_used_{CROSS_CARDS_MIN}min"] = int(cross_card_series.iloc[-1])

    # Geographical Features
    geo_df = df[["user_id", "card", "merchant_state", "merchant_city"]].copy()
    df["card_is_new_state"] = compute_is_new_state(geo_df, 'card', ['user_id', 'card'])
    df["card_is_new_city"] = compute_is_new_city(geo_df, 'card', ['user_id', 'card'])
    df["user_is_new_state"] = compute_is_new_state(geo_df, 'user', ['user_id'])
    df["user_is_new_city"] = compute_is_new_city(geo_df, 'user', ['user_id'])
    
    # Online Transactions Flag
    df["is_online"] = (df["use_chip"] == "online").astype("int8")

    # Cold-Start Handling
    df = apply_cold_start_values(df, request.app.state.cold_start)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Return current txn
    return df.iloc[[-1]][request.app.state.feature_cols]


@router.post("/predict", response_model=PredictionOutput)
@limiter.limit(RATE_LIMIT, exempt_when=is_localhost)
def predict_transaction(transaction: TransactionInput, request: Request):
    # Scoring one txn received from POST /predict

    logger.info(f"Rate Limit for external IP : {RATE_LIMIT}")
    logger.info(f"Request from : {request.client.host} | exempt : {is_localhost(request)}")

    start = time.time()

    try:
        features = build_features(transaction, request)

        fraud_prob = float(
            request.app.state.model.predict_proba(features)[:, 1][0]
        )

        risk_tier = compute_risk_tier(fraud_prob)
        exposure = compute_expected_exposure(
            fraud_prob, 
            transaction.amount,
            risk_tier)
        is_alert = fraud_prob >= request.app.state.threshold

        # for risk_tier = "high" or "critical" : Computing SHAP
        shap_explainability = None
        if risk_tier in (RiskTier.high, RiskTier.critical):
            try:
                shap_explainability = explain_transaction(request.app.state.model, features)
            except Exception as e:
                logger.warning(f"SHAP failed for {transaction.transaction_id} : {e}")
        
        latency_ms = round((time.time() - start) * 1000, 2)

        if transaction.transaction_id.startswith("USER_"):
            source = "user"
        elif transaction.transaction_id.startswith("DEMO_"):
            source = "demo"
        else:
            source = None

        if source:
            insert_scored_transaction({
                "transaction_id" : transaction.transaction_id,
                "user_id" : transaction.user_id,
                "card" : transaction.card,
                "timestamp" : transaction.timestamp,
                "amount" : transaction.amount,
                "merchant_name" : transaction.merchant_name,
                "merchant_city" : transaction.merchant_city,
                "merchant_state" : transaction.merchant_state,
                "mcc" : transaction.mcc,
                "use_chip" : transaction.use_chip,
                "errors" : transaction.errors or None,
                "source" : source,
            })

        insert_prediction({
            "transaction_id" : transaction.transaction_id,
            "model_version" : MODEL_VERSION,
            "fraud_probability" : round(fraud_prob, 6),
            "risk_tier" : risk_tier,
            "expected_exposure" : exposure,
            "inference_latency" : latency_ms,
            "is_alert" : is_alert
        })

        logger.info(
            f"{transaction.transaction_id} | "
            f"prob : {fraud_prob:.6f} | "
            f"tier : {risk_tier} | "
            f"alert : {is_alert} | "
            f"latency : {latency_ms} ms"
        )

        return PredictionOutput(
            transaction_id= transaction.transaction_id,
            fraud_probability= round(fraud_prob, 6),
            risk_tier= risk_tier,
            expected_exposure= exposure,
            is_alert= is_alert,
            inference_latency= latency_ms,
            shap_explainability= shap_explainability
        )
    
    except Exception as e:
        logger.error(f"Prediction failed for {transaction.transaction_id} : {e}")
        raise HTTPException(status_code=500, detail=str(e))