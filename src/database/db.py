import os
import yaml
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

# Loading config
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

db = config["postgres"]
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:"
    f"{db['port']}/{db['database']}",
    pool_size = db['pool_size'],
    max_overflow = db['max_overflow'],
    pool_pre_ping=True
)

def insert_prediction(records: dict) -> None:
    # Called by POST /predict for every txn

    with engine.connect() as conn:
        conn.execute(
            text("""
                 INSERT INTO predictions (
                 transaction_id, 
                 model_version, 
                 fraud_probability,
                 risk_tier, 
                 expected_exposure, 
                 inference_latency, 
                 is_alert
                 
                 ) VALUES (
                 :transaction_id, 
                 :model_version, 
                 :fraud_probability,
                 :risk_tier, 
                 :expected_exposure, 
                 :inference_latency, 
                 :is_alert
                 )
            """),
            records,
        )
        conn.commit()


def get_recent_alerts(limit: int = 50) -> list[dict]:
    # Called by GET /alerts endpoint

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT
                 p.transaction_id,
                 p.predicted_at,
                 p.model_version,
                 p.fraud_probability,
                 p.risk_tier,
                 p.expected_exposure,
                 p.inference_latency,
                 t.user_id,
                 t.card,
                 t.amount,
                 t.merchant_city,
                 t.merchant_state,
                 t.use_chip,
                 t.timestamp AS txn_timestamp
                FROM predictions p
                LEFT JOIN transactions t
                 ON p.transaction_id = t.transaction_id
                WHERE p.is_alert = TRUE
                ORDER BY p.predicted_at DESC
                LIMIT :limit
            """),
            {"limit": limit}
        ).fetchall()

    return [dict(row._mapping) for row in rows]


def get_metrics() -> dict:
    # Called by GET /metrics endpoint

    with engine.connect() as conn:
        totals = conn.execute(
            text("""
                SELECT
                 COUNT(*) AS total_predictions,
                 SUM(CASE WHEN is_alert THEN 1 ELSE 0 END) AS total_alerts,
                 ROUND(AVG(fraud_probability)::NUMERIC, 6) AS avg_fraud_prob,
                 ROUND(AVG(expected_exposure)::NUMERIC, 2) AS avg_exposure,
                 ROUND(AVG(NULLIF(inference_latency, 0))::NUMERIC, 2) AS avg_latency_ms
                FROM predictions
            """)
        ).fetchone()

        tier_rows = conn.execute(
            text("""
                SELECT risk_tier, COUNT(*) AS count
                FROM predictions
                GROUP BY risk_tier
                ORDER BY count DESC
            """)
        ).fetchall()

        alert_over_time = conn.execute(
            text("""
                SELECT 
                 DATE_TRUNC('day', 
                    predicted_at -
                    CASE
                        WHEN EXTRACT(DAY FROM predicted_at) < 15
                        THEN (EXTRACT(DAY FROM predicted_at) - 1) * INTERVAL '1 day'
                        ELSE (EXTRACT(DAY FROM predicted_at) - 15) * INTERVAL '1 day'
                    END
                 ) AS month, 
                 COUNT(*) AS total,
                 SUM(CASE WHEN is_alert THEN 1 ELSE 0 END) AS alerts
                FROM predictions
                GROUP BY 1
                ORDER BY 1 ASC
            """)
        ).fetchall()

        return {
            "total_predictions": totals.total_predictions or 0,
            "total_alerts" : totals.total_alerts or 0,
            
            "alert_rate" : round(
                totals.total_alerts / totals.total_predictions * 100, 4
            ) if totals.total_predictions > 0 else 0.0,
            
            "avg_fraud_prob" : float(totals.avg_fraud_prob or 0),
            "avg_exposure" : float(totals.avg_exposure or 0),
            "avg_latency_ms" : float(totals.avg_latency_ms or 0),
            
            "risk_tiers" : {
                row.risk_tier: row.count for row in tier_rows
            },
            
            "alert_over_time" : [{
                    "month" : row.month.isoformat(),
                    "total" : row.total,
                    "alerts" : row.alerts
                }
                for row in alert_over_time
            ]
        }
    

def get_transaction(transaction_id: str) -> dict | None:
    # Called by POST /predict for raw txns

    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT *
                FROM transactions
                WHERE transaction_id = :transaction_id
            """),
            {"transaction_id": transaction_id}
        ).fetchone()

    return dict(row._mapping) if row else None


def insert_scored_transaction(record: dict) -> None:
    # Inserts raw transaction inputs into scored_transactions table
    # Called by POST /predict for demo and user transactions

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO scored_transactions(
                transaction_id, user_id, card, timestamp, amount,
                merchant_name, merchant_city, merchant_state, mcc,
                use_chip, errors, source
            ) VALUES (
                :transaction_id, :user_id, :card, :timestamp, :amount,
                :merchant_name, :merchant_city, :merchant_state, :mcc,
                :use_chip, :errors, :source
            )
            ON CONFLICT (transaction_id) DO NOTHING
        """), record
        )
        conn.commit()


def get_batch_scored_transactions(transaction_ids: list) -> dict:
    # Fetches data from stored_transactions table

    if not transaction_ids:
        return {}

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT 
                    transaction_id, user_id, card, amount, 
                    merchant_city, merchant_state, use_chip, timestamp
                FROM scored_transactions
                WHERE transaction_id = ANY(:transaction_ids)
            """),
            {"transaction_ids" : transaction_ids}
        ).fetchall()
    
    return {
        row[0] : {
            "transaction_id" : row[0],
            "user_id" : row[1], 
            "card" : row[2], 
            "amount" : row[3], 
            "merchant_city" : row[4], 
            "merchant_state" : row[5], 
            "use_chip" : row[6], 
            "timestamp" : row[7]
        }
        for row in rows
    }


def log_audit_event(
    event_type: str, 
    details: str, 
    transaction_id: str = None
) -> None:
    # Logging into audit_logs table

    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO audit_logs (transaction_id, event_type, details)
                VALUES (:transaction_id, :event_type, :details)
            """),
            {
                "transaction_id": transaction_id,
                "event_type" : event_type,
                "details" : details
            }
        )
        conn.commit()