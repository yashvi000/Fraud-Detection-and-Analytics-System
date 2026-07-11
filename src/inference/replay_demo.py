import pandas as pd
from pathlib import Path
import logging
import yaml
import os
import time
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Setting up logger
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "replay_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Loading config and paths
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

FEATURES_PATH = PROJECT_ROOT / config["data"]["features_path"]

V1_TEST_START = config["splits"]["v1_test_start"]
V1_TEST_END = config["splits"]["v1_test_end"]

SAMPLE_SIZE = config["inference"]["demo"]["sample_size"]
DELAY_SECONDS = config["inference"]["demo"]["delay_seconds"]
BATCH_LOG_IN = config["inference"]["demo"]["batch_log_in"]

load_dotenv()

API_HOST = os.getenv("API_HOST")
API_PORT = os.getenv("API_PORT")
API_URL = f"http://{API_HOST}:{API_PORT}/predict"


def main():
    logger.info(f"Loading V1 test transactions ({V1_TEST_START}-{V1_TEST_END}) ...")

    df = pd.read_parquet(FEATURES_PATH)
    year = df["timestamp"].dt.year
    df = (
        df[(year >= V1_TEST_START) & (year <= V1_TEST_END)]
        .reset_index(drop=True)
    )

    if SAMPLE_SIZE:
        df = df.head(SAMPLE_SIZE)
        logger.info(f"Sample size : {SAMPLE_SIZE:,} transactions")

    logger.info(f"Transactions to send : {len(df):,}")
    logger.info(f"API URL : {API_URL}")
    logger.info(f"Delay per transaction : {DELAY_SECONDS} s")

    total_sent = 0
    total_alerts = 0
    errors = 0

    with requests.Session() as session:
        for i, row in df.iterrows():

            payload = {
                "transaction_id" : f"DEMO_{i + 1:012d}",
                "user_id" : int(row["user_id"]),
                "card" : int(row["card"]),
                "timestamp" : row["timestamp"].isoformat(),
                "amount" : float(row["amount"]),
                "use_chip" : str(row["use_chip"]),
                "merchant_name" : int(row["merchant_name"]),
                "merchant_city" : str(row["merchant_city"]),
                "merchant_state" : str(row["merchant_state"]),
                "mcc" : int(row["mcc"]),
                "errors" : str(row.get("errors", "None"))
            }

            try:
                response = session.post(API_URL, json=payload, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    total_sent += 1
                    total_alerts += int(result["is_alert"])
                
                else:
                    errors += 1
                    logger.warning(
                        f"DEMO_{i + 1:012d} | "
                        f"status : {response.status_code} | "
                        f"{response.text[:100]}"
                    )
            
            except requests.exceptions.RequestException as e:
                errors += 1
                logger.error(f"Request failed for DEMO_{i + 1:012d} : {e}")

        
            if (i + 1) % BATCH_LOG_IN == 0:
                logger.info(
                    f"Progress : {total_sent:,} / {len(df):,} | "
                    f"Alerts : {total_alerts:,} | "
                    f"Errors : {errors:,}"
                )
            
            time.sleep(DELAY_SECONDS)

    logger.info("Replay Demo Completed")
    logger.info(f"Sent : {total_sent:,}")
    logger.info(f"Alerts : {total_alerts:,}")
    logger.info(f"Errors : {errors:,}")

    if total_sent > 0:
        logger.info(f"Alert Rate : {total_alerts / total_sent * 100:.4f}%")
    

if __name__ == "__main__":
    main()