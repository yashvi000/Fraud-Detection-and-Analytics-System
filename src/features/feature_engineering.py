import pandas as pd
import yaml
from pathlib import Path
import logging
import gc

from .feature_utils import (
    compute_temp_features,
    compute_time_since_last_txn,
    compute_mcc_encoding,
    apply_mcc_encoding,
    compute_velocity_features,
    compute_spend_features,
    compute_zscore,
    compute_is_new_merchant,
    compute_cross_card_features,
    compute_is_new_state,
    compute_is_new_city,
    compute_cold_start_values,
    apply_cold_start_values
)

# Loading config and paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PROCESSED_PATH = PROJECT_ROOT / config["data"]["processed_path"]
FEATURES_PATH = PROJECT_ROOT / config["data"]["features_path"]
TRAIN_FEATURES_PATH = PROJECT_ROOT / config["data"]["train_features_path"]

VELOCITY_WINDOWS_MIN = config["features"]["velocity_windows_min"]
BASELINE_WINDOW_DAYS = config["features"]["baseline_window_days"]
CROSS_CARDS_MIN = config["features"]["cross_cards_min"]

V1_TRAIN_END = config["splits"]["v1_train_end"]

CHECKPOINT_PATH = PROJECT_ROOT / "data" / "features" / "checkpoint.parquet"

# Loading logger
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "feature_engineering.log"

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

def main():

    # Checkpoint 1
    logger.info("-"*5 + " Checkpoint 1 begins " + "-"*5)

    # Loading data
    logger.info(f"Loading : {PROCESSED_PATH}")

    df = pd.read_parquet(PROCESSED_PATH, columns=[
        'user_id', 'card', 'timestamp', 'amount', 'is_refund',
        'mcc', 'use_chip', 'merchant_name', 'merchant_city', 
        'merchant_state', 'is_fraud', 'is_high_value', 'error_bad_cvv', 
        'error_bad_expiration', 'error_bad_card', 'error_bad_pin'
        ]
    )
    
    logger.info(f"Loaded {len(df):,} rows | {df['user_id'].nunique():,} users")


    # Temporal Features
    logger.info("Computing Temporal Features ...")
    
    df = compute_temp_features(df)
    df["minutes_since_last_txn"] = compute_time_since_last_txn(df)
    logger.info("Computed Temporal Features")


    # MCC Encoding (training data)
    logger.info("Computing MCC Frequency Encoding ...")

    train_mask = df["timestamp"].dt.year <= V1_TRAIN_END
    mcc_freq = compute_mcc_encoding(df[train_mask])
    df["mcc_frequency"] = apply_mcc_encoding(df, mcc_freq)


    # Card Features
    logger.info("Computing Card Velocity and Spend Features ...")

    df = df.sort_values(["user_id", "card", "timestamp"]).reset_index(drop=True)
    group_cards = df.groupby(["user_id", "card"], sort=False)

    for window in VELOCITY_WINDOWS_MIN:
        col_name = f"card_txn_count_{window}min"
        df[col_name] = (
            group_cards[['timestamp', "is_refund"]]
            .apply(lambda x: compute_velocity_features(x, window_min=window),
                   include_groups=False)
            .reset_index(level=[0, 1], drop=True)
            .astype("float32")
        )
        logger.info(f"Computed '{col_name}'")

    for window in BASELINE_WINDOW_DAYS:
        col_name = f"card_txn_count_{window}d"
        df[col_name] = (
            group_cards[['timestamp', "is_refund"]]
            .apply(lambda x: compute_velocity_features(x, window_min=window * 1440),
                   include_groups=False)
            .reset_index(level=[0, 1], drop=True)
            .astype("float32")
        )
        logger.info(f"Computed '{col_name}'")

    for window in BASELINE_WINDOW_DAYS:
        card_spend_features = (
            group_cards
            .apply(
                lambda x: compute_spend_features(x, 'card', window), 
                include_groups=False
            )
            .reset_index(level=[0, 1], drop=True)
            .astype("float32")
        )
        df[card_spend_features.columns] = card_spend_features
        logger.info(f"Computed Card Spend Features ({window} days)")


    # User Features
    logger.info("Computing User Velocity and Spend Features ...")

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    group_users = df.groupby(["user_id"], sort=False)

    for window in VELOCITY_WINDOWS_MIN:
        col_name = f"user_txn_count_{window}min"
        df[col_name] = (
            group_users[['timestamp', "is_refund"]]
            .apply(lambda x: compute_velocity_features(x, window_min=window),
                   include_groups=False)
            .reset_index(level=0, drop=True)
            .astype("float32")
        )
        logger.info(f"Computed '{col_name}'")

    for window in BASELINE_WINDOW_DAYS:
        col_name = f"user_txn_count_{window}d"
        df[col_name] = (
            group_users[['timestamp', "is_refund"]]
            .apply(lambda x: compute_velocity_features(x, window_min=window * 1440),
                   include_groups=False)
            .reset_index(level=0, drop=True)
            .astype("float32")
        )
        logger.info(f"Computed '{col_name}'")

    for window in BASELINE_WINDOW_DAYS:
        user_spend_features = (
            group_users
            .apply(
                lambda x: compute_spend_features(x, 'user', window),
                include_groups=False
            )
            .reset_index(level=0, drop=True)
            .astype("float32")
        )
        df[user_spend_features.columns] = user_spend_features
        logger.info(f"Computed User Spend Features ({window} days)")


    # Z-Score Features
    logger.info("Computing Z-Score Features ...")

    for window in BASELINE_WINDOW_DAYS:
        df[f"card_amount_zscore_{window}d"] = compute_zscore(df, "card", window)
        logger.info(f"Computed card-level amount z-score ({window} days)")
        
        df[f"user_amount_zscore_{window}d"] = compute_zscore(df, "user", window)
        logger.info(f"Computed user-level amount z-score ({window} days)")

    # Dropping weak features
    drop_cols = [
        "card_spend_mean_365d", 
        "card_spend_std_365d", 
        "user_spend_mean_365d", 
        "user_spend_std_365d"
    ]
    df = df.drop(columns=drop_cols)

    # Writing to Checkpoint 1
    logger.info("Writing Checkpoint 1 ...\n")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CHECKPOINT_PATH, index=False, compression="zstd")

    del group_cards, group_users, card_spend_features, user_spend_features, train_mask, df
    gc.collect()   # Forced garbage collection


    # Checkpoint 2
    logger.info("-"*5 + " Checkpoint 2 begins " + "-"*5)
    df = pd.read_parquet(CHECKPOINT_PATH)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Merchant Familiarity Features
    logger.info("Computing Merchant Familiarity Features ...")
    merchant_df = df[["user_id", "card", "merchant_name"]].copy()

    df["card_is_new_merchant"] = compute_is_new_merchant(merchant_df, 'card', ['user_id', 'card'])
    logger.info("Computed Card-level 'is_new_merchant' ")

    df["user_is_new_merchant"] = compute_is_new_merchant(merchant_df, 'user', ['user_id'])
    logger.info("Computed User-level 'is_new_merchant' ")

    del merchant_df
    gc.collect()   # Forced garbage collection

    # Cross-Card Level Features
    logger.info("Computing Cross-Card Level Features ...")

    cross_card_df = df[["user_id", "card", "timestamp"]].copy()
    cross_card_df = cross_card_df.groupby("user_id", sort=False)

    df[f"distinct_cards_used_{CROSS_CARDS_MIN}min"] = (
        compute_cross_card_features(cross_card_df, CROSS_CARDS_MIN)
        .fillna(0)
        .astype("int8")
    )
    logger.info(f"Computed 'distinct_cards_used{CROSS_CARDS_MIN}min'")
    
    del cross_card_df
    gc.collect()   # Forced garbage collection


    # Geographical Features
    logger.info("Computing Geographical Features ...")

    geo_card_df = df[["user_id", "card", "merchant_state", "merchant_city"]].copy()
    geo_user_df = df[["user_id", "merchant_state", "merchant_city"]].copy()

    df["card_is_new_state"] = compute_is_new_state(geo_card_df, 'card', ['user_id', 'card'])
    logger.info(f"Computed 'card_is_new_state'")

    df["card_is_new_city"] = compute_is_new_city(geo_card_df, 'card', ['user_id', 'card'])
    logger.info(f"Computed 'card_is_new_city'")

    df["user_is_new_state"] = compute_is_new_state(geo_user_df, 'user', ['user_id'])
    logger.info(f"Computed 'user_is_new_state'")

    df["user_is_new_city"] = compute_is_new_city(geo_user_df, 'user', ['user_id'])
    logger.info(f"Computed 'user_is_new_city'")

    del geo_card_df, geo_user_df
    gc.collect()   # Forced garbage collection

    # Online Transactions Flag
    df["is_online"] = (df["use_chip"] == "online").astype("int8")
    logger.info("Computed 'is_online'")


    # Cold-Start Handling
    logger.info("Computing Cold-Start fill values ...")
    train_mask = df["timestamp"].dt.year <= V1_TRAIN_END

    fill_values = compute_cold_start_values(
        df[train_mask],
        BASELINE_WINDOW_DAYS,
        VELOCITY_WINDOWS_MIN
    )

    df = apply_cold_start_values(df, fill_values)
    logger.info("Applied Cold-Start fill values")
    del train_mask

    logger.info("Sorting by timestamp for chronological order ...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Saving to FEATURES_PATH
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(FEATURES_PATH, index=False, compression="zstd")
    logger.info(f"Saved : {FEATURES_PATH} | {len(df):,} rows | {df.shape[1]} columns")

    # Saving Training Features
    train_df = df[df["timestamp"].dt.year <= V1_TRAIN_END]
    train_df.to_parquet(TRAIN_FEATURES_PATH, index=False, compression="zstd")
    logger.info(f"Saved : {TRAIN_FEATURES_PATH} | {len(train_df):,} rows | {train_df.shape[1]} columns")
    del train_df

    # Deleting Checkpoint
    CHECKPOINT_PATH.unlink()
    logger.info("Checkpoint deleted\n")

    # Validation
    logger.info(f"Rows : {len(df):,}")
    logger.info(f"Columns : {df.shape[1]}")
    logger.info(f"Nulls : {df.isna().sum().sum():,}")
    logger.info(f"Fraud Rate : {df['is_fraud'].mean():.6f}\n\n\n")


if __name__ == "__main__":
    main()