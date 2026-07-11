import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

VELOCITY_MIN_PERIODS = config["features"]["velocity_min_periods"]
SPEND_MIN_PERIODS = config["features"]["spend_min_periods"]

MCC_ENCODING_PATH = PROJECT_ROOT / "models" / "artifacts" / "mcc_encoding.pkl"
COLD_START_PATH = PROJECT_ROOT / "models" / "artifacts" / "cold_start_fill_values.pkl"

# Loading logger
logger = logging.getLogger(__name__)

# Temporal Features using timestamp
def compute_temp_features(df: pd.DataFrame) -> pd.DataFrame:
    
    hour = df["timestamp"].dt.hour.astype("int8")  # 0-23 hours
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")  # 0(Monday) - 6(Sunday)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")  # 1- weekend, 0- not weekend
    df["is_night"] = (
        (hour >= 22) | (hour <= 5)   # 1- night, 0- not night
    ).astype("int8")

    # Cyclic Encoding (to represent hour 23:00 and hour 0:00 being close)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")

    del hour
    return df

# Time since last transaction (per user_id, per card)
def compute_time_since_last_txn(df: pd.DataFrame) -> pd.Series:

    group = df.sort_values(
        ["user_id", "card", "timestamp"]
    ).copy()
    
    last_timestamp = (
        group.groupby(["user_id", "card"])["timestamp"]
        .shift(1)   # doesn't include current transaction
    )

    minutes_since_last_txn = (
        (df["timestamp"] - last_timestamp)
        .dt.total_seconds() / 60
    ).fillna(-1).astype("float32")    # 1st transaction has had '-1' minutes since its last txn

    return minutes_since_last_txn

# MCC Encoding for training data
def compute_mcc_encoding(train_df: pd.DataFrame) -> dict:
    
    mcc_freq = (
        train_df["mcc"]
        .value_counts(normalize=True)
        .to_dict()
    )

    # Saving MCC Encoding
    MCC_ENCODING_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mcc_freq, MCC_ENCODING_PATH)

    logger.info(f"MCC Encoding computed : {len(mcc_freq)} unique MCC Codes")
    logger.info(f"MCC Encoding saved to path : {MCC_ENCODING_PATH}")

    return mcc_freq

# Applying computed MCC Encodings
def apply_mcc_encoding(df: pd.DataFrame, mcc_freq: dict) -> pd.Series:
    
    mcc_frequency_encoding = (
        df["mcc"]
        .map(mcc_freq)
        .fillna(0)              # unknowm mcc -> 0
        .astype("float32")
    )

    return mcc_frequency_encoding

# Loading MCC Encodings from disc
def load_mcc_encoding() -> dict:

    if not MCC_ENCODING_PATH.exists():
        raise FileNotFoundError(
            f"MCC Encoding not found at {MCC_ENCODING_PATH}"
        )

    return joblib.load(MCC_ENCODING_PATH)

# Velocity features over last 'window_min' minutes
def compute_velocity_features(group: pd.DataFrame, window_min: int) -> pd.Series:

    idx = group.index

    # Excluding refunds
    velocity = (
        pd.Series(
            np.where(group["is_refund"].to_numpy() == 0, 1.0, np.nan),
            index=group["timestamp"]
        )
        .shift(1)
        .rolling(f"{window_min}min", min_periods=VELOCITY_MIN_PERIODS)
        .count()
    )

    velocity.index = idx
    return velocity

# Spend features over last 'window' days for prefix = 'card' or 'user'
def compute_spend_features(group: pd.DataFrame, prefix: str, window) -> pd.DataFrame:
    
    amount = group.set_index("timestamp")["amount"]
    refund_mask = group["is_refund"].to_numpy() == 0   # Excluding refunds
    amount = amount.where(refund_mask)
    
    rolled = (
        amount
        .shift(1)
        .rolling(f"{window}D", min_periods=SPEND_MIN_PERIODS)
    )
    
    features = {
        f"{prefix}_spend_mean_{window}d" : rolled.mean().values,
        f"{prefix}_spend_std_{window}d" : rolled.std().values,
    }
    
    return pd.DataFrame(features, index=group.index)

# Fraud Signal Check for different features 
def compute_fraud_signal(df: pd.DataFrame, cols: list, is_zscore=False) -> pd.DataFrame:

    fraud_df = (
        df.groupby("is_fraud")[cols]
        .mean().T
        .rename(columns={0: "Non_Fraud_avg", 1: "Fraud_avg"})
    )

    fraud_df['signal'] = fraud_df['Fraud_avg'] > fraud_df['Non_Fraud_avg']

    if is_zscore:
        fraud_df["difference"] = (
            fraud_df["Fraud_avg"] - fraud_df["Non_Fraud_avg"]
        ).round(2)

        return fraud_df.reset_index(names='feature')

    fraud_df["multiplier"] = (
            fraud_df["Fraud_avg"] / 
            fraud_df["Non_Fraud_avg"].replace(0, 0.0001)
        ).round(2)
    
    return fraud_df.reset_index(names='feature')

# Z-Score of amount over 'window' days
def compute_zscore(df: pd.DataFrame, prefix: str, window) -> pd.DataFrame:
    
    mean = f"{prefix}_spend_mean_{window}d"
    std = f"{prefix}_spend_std_{window}d"
    
    zscore = (df['amount'] - df[mean]) / df[std].replace(0, 1)
    return zscore.clip(-10, 10).astype("float32")

# Merchant Familiarity (1- new merchant, 0- familiar merchant)
def compute_is_new_merchant(df: pd.DataFrame, prefix: str, cols: list) -> pd.Series:
    
    return (
        df.groupby(cols + ["merchant_name"])
        .cumcount()
        .eq(0)
        .astype("int8")
        .rename(f"{prefix}_is_new_merchant")
    )

# Number of cards a user has used in last 'window_min' minutes
def compute_cross_card_features(df: pd.DataFrame, window_min: int) -> pd.Series:
    result = []
    cutoff = np.timedelta64(window_min, "m")

    for _, group in df:

        times = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        cards = group["card"].to_numpy()
        counts = np.zeros(len(group), dtype=np.int8)

        for i in range(1, len(group)):
            left = np.searchsorted(times, times[i] - cutoff, side="left") 
            counts[i] = len(np.unique(cards[left:i]))

        result.append(pd.Series(
            counts, 
            index=group.index, 
            name=f"distinct_cards_used_{window_min}min",
            dtype="int8"
            )
        )
    
    return pd.concat(result).sort_index()

# New state or not (1- new state, 0- familiar state & online)
def compute_is_new_state(df: pd.DataFrame, prefix: str, cols: list) -> pd.Series:
    
    result = pd.Series(0, index=df.index, dtype="int8")

    mask = df["merchant_state"].ne("ONLINE")  # Excluding online payments

    result.loc[mask] = (
        df.loc[mask]
        .groupby(cols + ["merchant_state"])
        .cumcount()
        .eq(0)
        .astype("int8")
    )

    return result.rename(f"{prefix}_is_new_state")

# New city or not (1- new city, 0- familiar city & online)
def compute_is_new_city(df: pd.DataFrame, prefix: str, cols: list) -> pd.Series:

    result = pd.Series(0, index=df.index, dtype="int8")

    mask = df["merchant_city"].ne("ONLINE")  # Excluding online payments
    
    result.loc[mask] = (
        df.loc[mask]
        .groupby(cols + ["merchant_city"])
        .cumcount()
        .eq(0)
        .astype("int8")
    )

    return result.rename(f"{prefix}_is_new_city")

# Cold-Start Handling values are decided
def compute_cold_start_values(
    train_df: pd.DataFrame, 
    baseline_window_days: list,
    velocity_windows_min: list,
    save_path : Path = COLD_START_PATH
) -> dict:

    # Excluding refunds from train mean and std
    train_spend = train_df.loc[train_df["is_refund"] == 0, "amount"]
    train_spend_mean = train_spend.mean()
    train_spend_std = train_spend.std()

    logger.info(f"Train spend mean : {train_spend_mean:.2f}")
    logger.info(f"Train spend std : {train_spend_std:.2f}")

    fill_values = {}

    for window in baseline_window_days:
        for prefix in ["card", "user"]:

            # Spend Features are filled with global training mean & std
            if f"{prefix}_spend_mean_{window}d" in train_df.columns:
                fill_values[f"{prefix}_spend_mean_{window}d"] = train_spend_mean
                fill_values[f"{prefix}_spend_std_{window}d"] = train_spend_std

            # Velocity Features (days) are filled with 0
            fill_values[f"{prefix}_txn_count_{window}d"] = 0

            # Z-Scores of Amount are filled with 0
            fill_values[f"{prefix}_amount_zscore_{window}d"] = 0

    # Velocity Features (mins) are filled with 0
    for window in velocity_windows_min:
        for prefix in ["card", "user"]:
            fill_values[f"{prefix}_txn_count_{window}min"] = 0

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fill_values, save_path)
        logger.info(f"Cold-Start fill values saved to : {save_path}")

    return fill_values

# Applying Cold-Start values
def apply_cold_start_values(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    
    # Skipping dropped columns
    fill_cols = {
        col: val
        for col, val in fill_values.items()
        if col in df.columns
    }

    df = df.fillna(fill_cols)
    return df

# Loading Cold-Start values from disc
def load_cold_start_values(save_path: Path) -> dict:

    if not save_path.exists():
        raise FileNotFoundError(
            f"Cold-Start fill values not found at {save_path}"
        )
    
    return joblib.load(save_path)


# Creating raw features that are not in incoming transaction (/predict)
def compute_raw_features(
    amount: float,
    errors: str,
    high_value_threshold: float
) -> dict:
    
    errors = errors or "None"

    return {
        "is_refund" : int(amount < 0),
        "is_high_value" : int(amount >= high_value_threshold),
        "error_bad_cvv" : int("Bad CVV" in errors),
        "error_bad_expiration" : int("Bad Expiration" in errors),
        "error_bad_card" : int("Bad Card" in errors),
        "error_bad_pin" : int("Bad Pin" in errors),
    }