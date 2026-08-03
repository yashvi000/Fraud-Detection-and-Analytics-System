import pandas as pd
import numpy as np
import gc
import os
from pathlib import Path
import logging
import yaml
import mlflow
import lightgbm as lgb
import joblib
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
        logging.FileHandler(LOG_DIR / "train_lgbm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Loading config and paths
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

FEATURES_PATH = PROJECT_ROOT / config["data"]["features_path"]
ARTIFACTS_PATH = PROJECT_ROOT / config["artifacts"]["model_dir"]

# Splits
V1_TRAIN_START = config["splits"]["v1_train_start"]
V1_TRAIN_END = config["splits"]["v1_train_end"]
V1_VAL_START = config["splits"]["v1_val_start"]
V1_VAL_END = config["splits"]["v1_val_end"]

RANDOM_STATE = config["random_state"]
SCALE_POS_WEIGHT = config["model"]["lgbm"]["baseline"]["scale_pos_weight"]
VAL_TUNE_SIZE = config["model"]["lgbm"]["val_tune_size"]

FEATURE_COLS = config["feature_cols"]

logger.info(f"Loading : {FEATURES_PATH}")
df = pd.read_parquet(FEATURES_PATH)
year = df["timestamp"].dt.year

v1_train = df[(year >= V1_TRAIN_START) & (year <= V1_TRAIN_END)]
v1_val = df[(year >= V1_VAL_START) & (year <= V1_VAL_END)]

del df
gc.collect()

x_train = v1_train[FEATURE_COLS].astype("float32")
y_train = v1_train["is_fraud"]

x_val = v1_val[FEATURE_COLS].astype("float32")
y_val = v1_val["is_fraud"]

del v1_train, v1_val
gc.collect()

logger.info(f"Train : {len(x_train):,} Rows | Val : {len(x_val):,} Rows")

# Tuned stratified val with oversampling
fraud_val_idx = x_val[y_val == 1].index
non_fraud_val_idx = x_val[y_val == 0].index

# Oversampling frauds
tune_val_fraud_count = min(len(fraud_val_idx), 1000)
tune_val_non_fraud_count = VAL_TUNE_SIZE - tune_val_fraud_count

tune_val_idx = np.concatenate([
    np.random.RandomState(RANDOM_STATE).choice(fraud_val_idx, tune_val_fraud_count, replace=False),
    np.random.RandomState(RANDOM_STATE).choice(non_fraud_val_idx, tune_val_non_fraud_count, replace=False)
])

x_val_tune = x_val.loc[tune_val_idx]
y_val_tune = y_val.loc[tune_val_idx]

logger.info(f"Early Stopping Val : {len(x_val_tune):,} Rows")
logger.info(f"Early Stopping Val Fraud Count : {y_val_tune.sum():,}")


# Training LightGBM
final_parameters = {
    "objective" : config["model"]["lgbm"]["baseline"]["objective"],
    "metric" : "auc",   # For Early Stopping
    "scale_pos_weight" : SCALE_POS_WEIGHT,
    "random_state" : RANDOM_STATE,
    "verbosity" : -1,
    "n_jobs" : 1,
    **config["model"]["lgbm"]["tuned"]
}

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

mlflow.end_run()
with mlflow.start_run(run_name="v1_final"):

    mlflow.set_tag("model_version", os.getenv("MODEL_VERSION"))
    mlflow.log_params(final_parameters)

    mlflow.log_params({
        "train_rows" : len(x_train),
        "val_rows" : len(x_val),
        "n_features" : len(FEATURE_COLS),
        "train_period" : f"{V1_TRAIN_START}-{V1_TRAIN_END}",
        "val_period" : f"{V1_VAL_START}-{V1_VAL_END}"
    })

    logger.info("Training V1 LightGBM ...")

    lgbm_v1 = lgb.LGBMClassifier(**final_parameters)
    lgbm_v1.fit(
        x_train, y_train,
        callbacks=[
            lgb.log_evaluation(period=-1)
        ]
    )

    trees_used = final_parameters["n_estimators"]
    logger.info(f"Trees Used : {trees_used}")
    mlflow.log_param("trees_used", trees_used)


    # Saving Artifacts
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    lgbm_path = ARTIFACTS_PATH / "lgbm_v1.pkl"

    joblib.dump(lgbm_v1, lgbm_path)
    mlflow.log_artifact(str(lgbm_path))

    config["artifacts"]["lgbm_v1_path"] = str(lgbm_path.relative_to(PROJECT_ROOT))
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    logger.info(f"Saved Artifacts : {lgbm_path.name}")