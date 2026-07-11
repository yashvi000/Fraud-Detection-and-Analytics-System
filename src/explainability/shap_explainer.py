import pandas as pd
import numpy as np
import shap
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Loading logger
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "shap_explainer.log"

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

ARTIFACTS_DIR = PROJECT_ROOT / config["artifacts"]["model_dir"]
SHAP_PLOT_PATH = PROJECT_ROOT / config["artifacts"]["shap_global_plot_path"]
GLOBAL_SHAP_SAMPLE = config["shap"]["global_shap_sample"]


def compute_global_shap(model, x_sample: pd.DataFrame) -> None:
    # Computed once and displayed in dashboard

    logger.info(f"Computing Global SHAP : {len(x_sample):,} Rows")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)

    # Returning Fraud values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    logger.info("Global SHAP values computed")

    SHAP_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values,
        x_sample,
        plot_type="bar",
        show=False,
        max_display=20
    )

    plt.title("Global SHAP Feature Importance", fontsize = 18)
    plt.gcf().set_size_inches(12, 10)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved : {SHAP_PLOT_PATH}")


def explain_transaction(model, transaction: pd.DataFrame) -> dict:
    # Computing Local SHAP for single flagged (critical, high risk) transactions

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transaction)

    # Returning Fraud values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_dict = dict(zip(transaction.columns, shap_values[0]))

    return dict(sorted(
        shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
    ))


def main():

    FEATURES_PATH = PROJECT_ROOT / config["data"]["features_path"]
    RANDOM_SEED = config["random_state"]

    V1_TRAIN_START = config["splits"]["v1_train_start"]
    V1_TRAIN_END = config["splits"]["v1_train_end"]

    FEATURE_COLS = config["feature_cols"]

    logger.info("Loading model ...")
    model = joblib.load(ARTIFACTS_DIR / "lgbm_v1.pkl")

    logger.info(f"Loading features : {FEATURES_PATH}")
    
    df = pd.read_parquet(FEATURES_PATH)
    year = df["timestamp"].dt.year

    v1_train = df[(year >= V1_TRAIN_START) & (year <= V1_TRAIN_END)]
    x_train = v1_train[FEATURE_COLS]

    # Stratified SHAP sample
    fraud_idx = x_train.index[v1_train["is_fraud"] == 1]
    non_fraud_idx = x_train.index[v1_train["is_fraud"] == 0]

    fraud_count = min(len(fraud_idx), GLOBAL_SHAP_SAMPLE // 2)
    non_fraud_count = GLOBAL_SHAP_SAMPLE - fraud_count

    sample_idx = np.concatenate([
        np.random.RandomState(RANDOM_SEED).choice(fraud_idx, fraud_count, replace=False),
        np.random.RandomState(RANDOM_SEED).choice(non_fraud_idx, non_fraud_count, replace=False)
    ])

    x_sample = x_train.loc[sample_idx]
    logger.info(f"SHAP Sample : {len(x_sample):,} Rows "
                f"({fraud_count:,} Frauds | {non_fraud_count:,} Non-Frauds)")
    
    compute_global_shap(model, x_sample)

if __name__ == "__main__":
    main()