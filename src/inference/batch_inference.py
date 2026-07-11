import pandas as pd
import yaml
import logging
import os
import joblib
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values
from pathlib import Path
from dotenv import load_dotenv

from src.risk.scoring import compute_risk_tier, compute_expected_exposure
from src.database.db import engine

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Loading logger
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "batch_inference.log"

logging.basicConfig(
    level= logging.INFO,
    format= "%(asctime)s | %(levelname)s | %(message)s",
    datefmt= "%d-%m-%Y %H:%M:%S",
    handlers= [
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Loading config and paths
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

FEATURES_PATH = PROJECT_ROOT / config["data"]["features_path"]
ARTIFACTS_DIR = PROJECT_ROOT / config["artifacts"]["model_dir"]

THRESHOLD = config["model"]["threshold"]
CHUNK_SIZE = config["inference"]["chunk_size"]
FEATURE_COLS = config["feature_cols"]

MODEL_VERSION = os.getenv("MODEL_VERSION")


def clear_predictions() -> None:
    
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE predictions;"))
        conn.commit()
    logger.info("Predictions table cleared")

def insert_predictions(records: list) -> None:
    
    with engine.connect() as conn:
        execute_values(
            conn.connection.cursor(),
            """
                INSERT INTO predictions (
                    transaction_id, model_version, fraud_probability,
                    risk_tier, expected_exposure, inference_latency, is_alert
                ) VALUES %s
            """,
            records,
            page_size=len(records)
        )
        conn.connection.commit()  


def main():

    logger.info("Loading model ...")
    model = joblib.load(ARTIFACTS_DIR / "lgbm_v1.pkl")

    V1_TEST_END = config["splits"]["v1_test_end"]

    logger.info(f"Loading features : {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    
    # For V1 only
    year = df["timestamp"].dt.year
    df = df[year <= V1_TEST_END].reset_index(drop=True)

    # Verifying chronological sorting order
    is_sorted = df["timestamp"].is_monotonic_increasing

    if not is_sorted:
        logger.warning("Features not in chronological order")
    logger.info(f"Loaded {len(df):,} V1 Rows (sorted chronologically)")
    
    df["transaction_id"] = [f"TXN_{i+1:012d}" for i in range(len(df))]

    clear_predictions()

    total_rows = len(df)
    total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_alerts = 0
    total_inserted = 0

    logger.info(f"Processing {total_rows:,} Rows | {total_chunks:,} Chunks |"
                f" Chunk Size : {CHUNK_SIZE:,}")
    
    for chunk_idx in range(total_chunks):
        
        start =  chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total_rows)
        chunk = df.iloc[start : end]

        fraud_prob = model.predict_proba(chunk[FEATURE_COLS])[:, 1]
        
        tiers = [compute_risk_tier(p) for p in fraud_prob]
        exposures = [
            compute_expected_exposure(float(p), float(a), t)
            for p, a, t in zip(fraud_prob, chunk["amount"].values, tiers)
        ]

        is_alerts = fraud_prob >= THRESHOLD

        records = list(zip(
            chunk["transaction_id"].values,
            [MODEL_VERSION] * len(chunk),
            fraud_prob.round(6).tolist(),
            tiers,
            exposures,
            [0.00] * len(chunk),
            is_alerts.tolist()
        ))

        insert_predictions(records)

        chunk_alerts = int(is_alerts.sum())
        total_alerts += chunk_alerts
        total_inserted += len(records)

        if (chunk_idx + 1) % 10 == 0 or chunk_idx == total_chunks - 1:
            logger.info(
                f"Progress : {total_inserted:,} / {total_rows:,} "
                f"({total_inserted / total_rows * 100:.2f}%) | "
                f"Alerts : {total_alerts:,}"
            )

    logger.info("Batch Inference completed")
    logger.info(f"Rows processed : {total_inserted:,}")
    logger.info(f"Total alerts : {total_alerts:,}")
    logger.info(f"Alert Rate : {total_alerts / total_inserted * 100:.4f}%")


if __name__ == "__main__":
    main()
    