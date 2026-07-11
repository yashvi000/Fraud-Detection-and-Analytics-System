import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
import yaml
import mlflow
import lightgbm as lgb
import joblib
from dotenv import load_dotenv

from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve, 
    f1_score, recall_score, precision_score
)

from .model_utils import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Setting up logger
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "evaluate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Loading config and paths
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

FEATURES_PATH = PROJECT_ROOT / config["data"]["features_path"]

# Splits
V1_VAL_START = config["splits"]["v1_val_start"]
V1_VAL_END = config["splits"]["v1_val_end"]
V1_TEST_START = config["splits"]["v1_test_start"]
V1_TEST_END = config["splits"]["v1_test_end"]

BEST_THRESHOLD = config["model"]["threshold"]
FP_COST = config["model"]["fp_cost"]
RECALL_FLOOR = config["model"]["recall_floor"]

FEATURE_COLS = config["feature_cols"]

# Loading Model
lgbm_path = PROJECT_ROOT / config["artifacts"]["lgbm_v1_path"]
logger.info(f"Loading model : {lgbm_path.name}")
lgbm_v1 = joblib.load(lgbm_path)

# Loading Data
logger.info(f"Loading : {FEATURES_PATH}")
df = pd.read_parquet(FEATURES_PATH)
year = df["timestamp"].dt.year

v1_val = df[(year >= V1_VAL_START) & (year <= V1_VAL_END)]
v1_test = df[(year >= V1_TEST_START) & (year <= V1_TEST_END)]

x_val = v1_val[FEATURE_COLS]
y_val = v1_val["is_fraud"]

x_test = v1_test[FEATURE_COLS]
y_test = v1_test["is_fraud"]

logger.info(f"Val : {len(x_val):,} Rows | Test : {len(x_test):,} Rows")

# Evaluating
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))
mlflow.set_tag("model_version", os.getenv("MODEL_VERSION"))

mlflow.end_run()
with mlflow.start_run(run_name="v1_evaluation"):
    
    logger.info("Evaluating Model ...")
    
    _, val_metrics = evaluate_model(lgbm_v1, x_val, y_val, BEST_THRESHOLD, "V1 Val")
    test_prob, test_metrics = evaluate_model(lgbm_v1, x_test, y_test, BEST_THRESHOLD, "V1 Test")

    mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    logger.info(f"Val PR-AUC : {val_metrics['pr_auc']:.5f}")
    logger.info(f"Test PR-AUC : {test_metrics['pr_auc']:.5f}")

    # Financial Impact
    logger.info("Calculating Financial Impact on V1 Test ...")
    y_test_np = y_test.to_numpy()
    test_fraud_mask = y_test_np == 1
    test_fraud_amounts = df.loc[x_test.index[test_fraud_mask], "amount"].to_numpy()

    test_fraud_prob = test_prob[test_fraud_mask]
    test_non_fraud_prob = test_prob[~test_fraud_mask]

    test_fraud_caught = int((test_fraud_prob >= BEST_THRESHOLD).sum())
    test_fraud_missed = int((test_fraud_prob < BEST_THRESHOLD).sum())
    test_false_alarms = int((test_non_fraud_prob >= BEST_THRESHOLD).sum())

    test_fn_loss = float(test_fraud_amounts[test_fraud_prob <   BEST_THRESHOLD].sum())
    test_fp_loss = test_false_alarms * FP_COST
    test_total_loss = test_fn_loss + test_fp_loss

    mlflow.log_metrics({
        "test_fraud_caught" : test_fraud_caught,
        "test_fraud_missed" : test_fraud_missed,
        "test_false_alarms" : test_false_alarms,
        "test_fn_loss" : round(test_fn_loss, 2),
        "test_fp_loss" : round(test_fp_loss, 2),
        "test_total_loss" : round(test_total_loss, 2),
    })

    logger.info(f"Fraud Caught : {test_fraud_caught:,} / {len(test_fraud_amounts):,} "
                f"({test_fraud_caught / len(test_fraud_amounts) * 100:.2f}%)")
    logger.info(f"False Alarms : {test_false_alarms:,}")
    logger.info(f"Total Loss : ${test_total_loss:,.2f}")
    