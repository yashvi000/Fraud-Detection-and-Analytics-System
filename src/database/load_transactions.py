import os
import yaml
import duckdb
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import time

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PROCESSED_PATH = PROJECT_ROOT / config["data"]["processed_path"]

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Connection to postgreSQL
CONNECTION = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
CONNECTION_STR = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

THREADS = config["duckdb"]["threads"]
MEMORY  = config["duckdb"]["memory_limit"]

def run_sql(engine, sql):
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()

def main():
    start_time = time.time()

    print(f"Loading parquet: {PROCESSED_PATH}")
    print(f"Into Postgres  : {DB_NAME}\n")

    engine = create_engine(CONNECTION_STR)

    # Dropping indexes before loading
    run_sql(engine, "DROP INDEX IF EXISTS idx_user_card_time;")
    run_sql(engine, "DROP INDEX IF EXISTS idx_timestamp;")
    print("Indexes dropped")

    # Truncating transactions table
    run_sql(engine, "TRUNCATE TABLE transactions;")
    print("transaction table truncated")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute(f"PRAGMA memory_limit='{MEMORY}'")

    # Attaching Postgres
    try:
        con.execute("INSTALL postgres;")
    except Exception:
        pass
    con.execute("LOAD postgres;")
    con.execute(f"ATTACH '{CONNECTION}' AS pgdb (TYPE POSTGRES);")
    
    con.execute(f"""
        INSERT INTO pgdb.transactions (
            transaction_id,
            user_id,
            card,
            timestamp,
            amount,
            is_refund,
            merchant_name,
            merchant_city,
            merchant_state,
            zip,
            mcc,
            use_chip,
            errors,
            error_bad_cvv,
            error_bad_expiration,
            error_bad_card,
            error_bad_pin,
            is_high_value,
            is_fraud
        )

        SELECT
            'TXN_' || LPAD(CAST
                (row_number() OVER (
                    ORDER BY timestamp
                ) AS VARCHAR), 12, '0') AS transaction_id,
            user_id,
            card,
            timestamp,
            amount,
            is_refund::BOOLEAN,
            merchant_name,
            merchant_city,
            merchant_state,
            zip,
            mcc,
            use_chip,
            CAST(errors AS VARCHAR) AS errors,
            error_bad_cvv::BOOLEAN,
            error_bad_expiration::BOOLEAN,
            error_bad_card::BOOLEAN,
            error_bad_pin::BOOLEAN,
            is_high_value::BOOLEAN,
            is_fraud::BOOLEAN
        FROM read_parquet('{str(PROCESSED_PATH)}');
    """)

    # Verify count
    count = con.execute("SELECT COUNT(*) FROM pgdb.transactions;").fetchone()[0]
    print("Data loaded into PostgreSQL")
    con.close()

    # Recreating indexes
    run_sql(engine, """
        CREATE INDEX IF NOT EXISTS idx_user_card_time
        ON transactions(user_id, card, timestamp);
    """)

    run_sql(engine, """
        CREATE INDEX IF NOT EXISTS idx_timestamp
        ON transactions(timestamp);
    """)
    print("Indexes recreated")

    end_time = time.time()
    elapsed = (end_time - start_time) / 60
    rows_per_sec = count / elapsed / 60

    print(f"\nTotal time taken : {elapsed:.2f} minutes")
    print(f"Rows loaded : {count:,}")
    print(f"Rows loaded per second : {rows_per_sec:,.0f}")

    # Logging in audits
    run_sql(engine, f"""
        INSERT INTO audit_logs (event_type, details)
        VALUES (
            'data_load', 
            'Loaded {count:,} rows in {elapsed:.2f} minutes ({rows_per_sec:,.0f} rows/sec)'
        );
    """)

if __name__ == "__main__":
    main()