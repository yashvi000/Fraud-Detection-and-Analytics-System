import pandas as pd
import yaml
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Loading config and paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RAW_PATH = PROJECT_ROOT / config["data"]["raw_path"]
PROCESSED_PATH = PROJECT_ROOT / config["data"]["processed_path"]
TEMP_PATH = PROCESSED_PATH.parent / "transactions_temp.parquet"

HIGH_VALUE_THRESHOLD = config["features"]["high_value_threshold"]

CHUNK_SIZE = 1_000_000

PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

if TEMP_PATH.exists():
    TEMP_PATH.unlink()

print(f"Input : {RAW_PATH}")
print(f"Output : {PROCESSED_PATH}")

DTYPES = {
    "User" : "int32",
    "Card" : "int16",
    "Year" : "int16",
    "Month" : "int8",
    "Day" : "int8",
    "Time" : "string",
    "Amount" : "string",
    "Use Chip" : "category",
    "Merchant Name" : "int64",
    "Merchant City" : "string",
    "Merchant State" : "string",
    "Zip" : "string",
    "MCC" : "int16",
    "Errors?" : "string",
    "Is Fraud?" : "string"
}

COLUMN_NAMES = {
    "User" : "user_id",
    "Card" : "card",
    "Amount" : "amount",
    "Use Chip" : "use_chip",
    "Merchant Name" : "merchant_name",
    "Merchant City" : "merchant_city",
    "Merchant State" : "merchant_state",
    "Zip" : "zip",
    "MCC" : "mcc",
    "Errors?" : "errors",
    "Is Fraud?" : "is_fraud"
}

# Cleaning per chunk
def cleaning_by_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.rename(columns=COLUMN_NAMES)

    # Parsing timestamp
    ts = (
        chunk["Year"].astype(str) + "-" +
        chunk["Month"].astype(str).str.zfill(2) + "-" +
        chunk["Day"].astype(str).str.zfill(2) + " " +
        chunk["Time"].astype(str).str.strip()
    )

    chunk["timestamp"] = pd.to_datetime(ts, format="%Y-%m-%d %H:%M", errors="coerce")
    missing = chunk["timestamp"].isna()
    if missing.any():
        chunk.loc[missing, "timestamp"] = pd.to_datetime(
            ts.loc[missing], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )

    chunk = chunk.drop(columns=["Year", "Month", "Day", "Time"])
    chunk = chunk.dropna(subset=["timestamp"])

    # Converting "amount" from str to float
    chunk["amount"] = pd.to_numeric(
        chunk["amount"]
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False),
        errors="coerce"
    ).astype("float64").round(2)
    chunk = chunk.dropna(subset=["amount"])

    # Encoding "is_fraud"
    chunk["is_fraud"] = (chunk["is_fraud"] == "Yes").astype("int8")   # 1- fraud, 0- not fraud

    # Refund flag (-ve amount = refund)
    chunk["is_refund"] = (chunk["amount"] < 0).astype("int8")  #1- refund, 0- not refund

    # Simplifying use_chip objects: "swipe", "chip", "online"
    chunk["use_chip"] = (
        chunk["use_chip"]
        .astype(str)
        .str.strip()
        .str.replace(" Transaction", "", regex=False)
        .str.lower()
    )
    chunk["use_chip"] = chunk["use_chip"].astype("category")

    # Missing location = "ONLINE"
    chunk["merchant_state"] = (
        chunk["merchant_state"]
        .fillna("ONLINE")
        .astype(str)
        .str.strip()
    )

    # Cleanig merchant_city
    chunk["merchant_city"] = (
        chunk["merchant_city"]
        .astype(str)
        .str.strip()
    )

    # Converting ZIP to string
    chunk["zip"] = (
        chunk["zip"]
        .fillna("0")
        .astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
    )

    # Filling missing Error rows
    chunk["errors"] = (
        chunk["errors"]
        .fillna("None")
        .astype(str)
        .str.strip()
        .str.rstrip(",")
    )

    # Flagging errors with highest fraud rates
    # 1- error present, 0- error not present
    chunk["error_bad_cvv"] = chunk["errors"].str.contains("Bad CVV", na=False).astype("int8")
    chunk["error_bad_expiration"] = chunk["errors"].str.contains("Bad Expiration", na=False).astype("int8")
    chunk["error_bad_card"] = chunk["errors"].str.contains("Bad Card Number", na=False).astype("int8")
    chunk["error_bad_pin"] = chunk["errors"].str.contains("Bad PIN", na=False).astype("int8")

    # Flagging high value transactions (1- high value, 0- not high value)
    chunk["is_high_value"] = (chunk["amount"] >= HIGH_VALUE_THRESHOLD).astype("int8")

    return chunk

def run_preprocessing():
    print(f"\nStarting preprocessing with chunk size : {CHUNK_SIZE:,}\n")

    writer = None
    total_rows = 0
    chunk_num = 0

    TRAIN_START_YEAR = config["splits"]["v1_train_start"]
    TRAIN_END_YEAR = config["splits"]["v1_train_end"]
    train_fraud = 0
    train_non_fraud = 0

    reader = pd.read_csv(
        RAW_PATH,
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    for chunk in reader:
        cleaned = cleaning_by_chunk(chunk)
        
        chunk_num += 1
        total_rows += len(cleaned)
        print(f"Preprocessing chunk {chunk_num} : {len(chunk):,} rows cleaned | Total rows preprocessed : {total_rows:,}")

        train_mask = (
            (cleaned["timestamp"].dt.year >= TRAIN_START_YEAR) &
            (cleaned["timestamp"].dt.year <= TRAIN_END_YEAR) &
            (cleaned["is_refund"] == 0)
        )

        fraud_count = int(cleaned.loc[train_mask, "is_fraud"].sum())
        non_fraud_count = int(train_mask.sum()) - fraud_count

        train_fraud += fraud_count
        train_non_fraud += non_fraud_count

        # Writing to TEMP_PATH
        table = pa.Table.from_pandas(cleaned, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(TEMP_PATH, table.schema)
        writer.write_table(table)

        del chunk, cleaned, table  # Freeing RAM

    if writer:
        writer.close()

    print(f"\nChunks written in temp file : {chunk_num}")
    print(f"Total rows written in temp file : {total_rows:,}")

    # Calculating scale_pos_weights from training data
    scale_pos_weight = train_non_fraud / max(train_fraud, 1)
    scale_pos_weight = round(scale_pos_weight, 4)

    print(f"\nTraining upto : {TRAIN_END_YEAR}")
    print(f"Fraud in train (except refund) : {train_fraud:,}")
    print(f"Non-fraud in train (except refund) : {train_non_fraud:,}")
    print(f"scale_pos_weight : {scale_pos_weight}")

    # Saving scale_pos_weight to config
    config["model"]["lgbm"]["scale_pos_weight"] = scale_pos_weight
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print("\nSaved scale_pos_weight to config.yaml")

    if PROCESSED_PATH.exists():
        PROCESSED_PATH.unlink()
    

    # Sorting using DuckDB
    print("\nSorting with DuckDB :")

    THREADS = config["duckdb"]["threads"]
    MEMORY = config["duckdb"]["memory_limit"]

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute(f"PRAGMA memory_limit='{MEMORY}'")
    con.execute(f"""
        COPY(
            SELECT *
            FROM read_parquet('{str(TEMP_PATH)}')
            ORDER BY user_id, card, timestamp
        ) TO '{str(PROCESSED_PATH)}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    print("Sorting completed")

    # Removing duplicates
    pre_dedup = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{str(PROCESSED_PATH)}')
    """).fetchone()[0]

    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{str(PROCESSED_PATH)}')
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY user_id, card, timestamp, amount
                ORDER BY timestamp
            ) = 1
        ) TO '{str(PROCESSED_PATH)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    post_dedup = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{str(PROCESSED_PATH)}')
    """).fetchone()[0]
    con.close()

    dupes_removed = pre_dedup - post_dedup
    print(f"\nDuplicates removed: {dupes_removed:,}")
    print(f"Final row count: {post_dedup:,}")

    # Validation checks using DuckDB
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute(f"PRAGMA memory_limit='{MEMORY}'")

    total_rows_out = con.execute(f"""
        SELECT COUNT(*)
        FROM read_parquet('{str(PROCESSED_PATH)}')
    """).fetchone()[0]

    ts_range = con.execute(f"""
        SELECT MIN(timestamp), MAX(timestamp)
        FROM read_parquet('{str(PROCESSED_PATH)}')
    """).fetchone()

    fraud_rate = con.execute(f"""
        SELECT AVG(is_fraud)
        FROM read_parquet('{str(PROCESSED_PATH)}')
    """).fetchone()[0]

    refund_rate = con.execute(f"""
        SELECT AVG(is_refund)
        FROM read_parquet('{str(PROCESSED_PATH)}')
    """).fetchone()[0]

    nulls = con.execute(f"""
        SELECT 
            COUNT(*) FILTER (WHERE user_id IS NULL) AS null_user,
            COUNT(*) FILTER (WHERE card IS NULL) AS null_card,
            COUNT(*) FILTER (WHERE timestamp IS NULL) AS null_timestamp,
            COUNT(*) FILTER (WHERE amount IS NULL) AS null_amount,
            COUNT(*) FILTER (WHERE is_fraud IS NULL) AS null_is_fraud,
            COUNT(*) FILTER (WHERE merchant_state IS NULL) AS null_merchant_state
        FROM read_parquet('{str(PROCESSED_PATH)}')
    """).fetchdf()

    use_chip_dist = con.execute(f"""
        SELECT use_chip, COUNT(*) AS count
        FROM read_parquet('{str(PROCESSED_PATH)}')
        GROUP BY use_chip
        ORDER BY count DESC
    """).fetchdf()

    error_counts = con.execute(f"""
        SELECT
            SUM(error_bad_cvv),
            SUM(error_bad_expiration),
            SUM(error_bad_card),
            SUM(error_bad_pin),
            SUM(is_high_value)
        FROM read_parquet('{str(PROCESSED_PATH)}')
    """).fetchone()

    con.close()
    
    print("\nSummary :-")
    print(f"Total rows : {total_rows_out:,}")
    print(f"Nulls : \n{nulls.to_string(index=False)}")

    print(f"\nTimestamp range : {ts_range[0]} to {ts_range[1]}")
    print(f"Fraud rate : {fraud_rate:.4f}")
    print(f"Refund rate : {refund_rate:.4f}")
    print(f"\nuse_chip values : \n{use_chip_dist.to_string(index=False)}")

    print(f"\nCount of error_bad_cvv : {int(error_counts[0])}")
    print(f"Count of error_bad_expiration : {int(error_counts[1])}")
    print(f"Count of error_bad_card : {int(error_counts[2])}")
    print(f"Count of error_bad_pin : {int(error_counts[3])}")
    print(f"High value transactions : {int(error_counts[4])}")

    TEMP_PATH.unlink(missing_ok=True)


# Entry Point
if __name__ == "__main__":
    run_preprocessing()