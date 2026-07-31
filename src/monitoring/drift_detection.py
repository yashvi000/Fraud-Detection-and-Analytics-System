# Evidently AI drift detection pipeline

import pandas as pd
import duckdb
import yaml
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Loading config and paths
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

THREADS = config["duckdb"]["threads"]
MEMORY_LIMIT = config["duckdb"]["memory_limit"]

REFERENCE_PATH = PROJECT_ROOT / config["monitoring"]["drift_reference_path"]
FEATURES_PATH = PROJECT_ROOT / config["data"]["features_path"]
DRIFT_REPORT_PATH = PROJECT_ROOT / config["monitoring"]["drift_report_path"]

FEATURE_COLS = config["feature_cols"]
SAMPLE_SIZE = config["monitoring"]["drift_sample_size"]
RANDOM_STATE = config["random_state"]
DATA_DRIFT_THRESHOLD = config["monitoring"]["data_drift_threshold"]

# Splits
V1_TRAIN_START= config["splits"]["v1_train_start"]
V1_TRAIN_END = config["splits"]["v1_train_end"]
DRIFT_YEAR = config["splits"]["drift_detection_year"]
V2_TEST_END = config["splits"]["v2_test_end"]

# Loading logger
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "drift_detection.log"

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


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    return con


def load_reference_data(sample_size: int = SAMPLE_SIZE) -> pd.DataFrame: 
    # Loads V1 training features for drift reference

    cols_sql = ", ".join(FEATURE_COLS)

    if REFERENCE_PATH.exists():
        source = REFERENCE_PATH
        where_clause = ""
        logger.info(f"Reference Source : {source.name}")
    
    elif FEATURES_PATH.exists():
        source = FEATURES_PATH
        where_clause = (
            f"WHERE YEAR(timestamp) >= {V1_TRAIN_START} "
            f"AND YEAR(timestamp) <= {V1_TRAIN_END}"
        )

        logger.warning(
            f"{REFERENCE_PATH.name} not found | "
            f"Falling back to {source.name} filtered to {V1_TRAIN_START}-{V1_TRAIN_END}"
        )
    
    else:
        raise FileNotFoundError(
            f"No reference data found.\n"
            f"Expected : {REFERENCE_PATH}\n"
            f"Fallback : {FEATURES_PATH}\n"
        )
    
    con = _con()

    try:
        df = con.execute(f"""
            SELECT {cols_sql}
            FROM read_parquet('{source}')
            {where_clause}
            USING SAMPLE {sample_size} ROWS (reservoir, {RANDOM_STATE})
        """).df()
    
    finally:
        con.close()

    logger.info(
        f"Reference Data loaded : {len(df):,} rows | "
        f"period : {V1_TRAIN_START}-{V1_TRAIN_END}"
    )
    return df


def load_current_data(
    year_start: int = DRIFT_YEAR,
    year_end: int = DRIFT_YEAR,
    sample_size: int = SAMPLE_SIZE
) -> pd.DataFrame:
    # Loads new data for drift detection (2015 only by default)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature parquet not found at {FEATURES_PATH}")
    
    cols_sql = ", ".join(FEATURE_COLS)
    con = _con()

    try:

        available = con.execute(f"""
            SELECT COUNT(*)
            FROM read_parquet('{FEATURES_PATH}')
            WHERE YEAR(timestamp) >= {year_start}
                AND YEAR(timestamp) <= {year_end}
        """).fetchone()[0]

        if available <= sample_size:
            query = f"""
                SELECT {cols_sql}
                FROM read_parquet('{FEATURES_PATH}')
                WHERE YEAR(timestamp) >= {year_start}
                    AND YEAR(timestamp) <= {year_end}
            """

            logger.info(
                f"Current data window has {available:,} rows | "
                f"Loading all (below sample threshold {sample_size:,} rows)"
            )
        
        else:
            query = f"""
                SELECT *
                FROM (
                    SELECT {cols_sql}
                    FROM read_parquet('{FEATURES_PATH}')
                    WHERE YEAR(timestamp) >= {year_start}
                        AND YEAR(timestamp) <= {year_end}
                )
                USING SAMPLE {sample_size} ROWS (reservoir, {RANDOM_STATE})
            """

            logger.info(
                f"Current data window has {available:,} rows | "
                f"Sampling ({sample_size:,} rows)"
            )
        
        df = con.execute(query).df()
    
    finally:
        con.close()

    period = (
        f"{year_start}-{year_end}"
        if year_start != year_end
        else str(year_start)
    )

    logger.info(
        f"Current Data loaded : {len(df):,} rows | "
        f"period : {period}"
    )
    return df


def parse_report_summary(report_dict: dict) -> dict:
    # Extracts clean drift summary from Evidently AI report dict

    drift_share = 0.00
    n_features = 0
    n_drifted = 0
    drifted_features = []

    try:
        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})

            # DatasetDriftMetric - overall summary of all features
            if "share_of_drifted_columns" in result:
                drift_share = float(result["share_of_drifted_columns"])
                n_features = int(result.get("number_of_columns", 0))
                n_drifted = int(result.get("number_of_drifted_columns", 0))
            
            # feature-wise drift result
            if "drift_by_columns" in result:
                drift_by_columns = result["drift_by_columns"]
                
                for col_name, col_result in drift_by_columns.items():
                    if col_result.get("drift_detected") and col_name not in drifted_features:
                        drifted_features.append(col_name)
    
    except Exception as e:
        logger.warning(f"Drift Report could not be parsed correctly : {e}")
    
    return {
        "drift_share": drift_share,
        "n_drifted": n_drifted,
        "n_features": n_features,
        "drifted_features": drifted_features,
    }


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    reference_period: str,
    current_period: str,
    output_path: Path = DRIFT_REPORT_PATH
) -> dict:
    """
    Runs Evidently AI DataDrift report comparing reference and current data

    Generates :
    1. HTML Report (shown in Streamlit dashboard)
    2. Summary dict (used in audit logging and dashboard KPIs)
    """

    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    logger.info("Running Evidently AI DataDrift Report ...")
    logger.info(f"Reference : {reference_period} | {len(reference_df):,} rows")
    logger.info(f"Current : {current_period} | {len(current_df):,} rows")

    ref = reference_df[FEATURE_COLS].astype("float64").reset_index(drop=True)
    curr = current_df[FEATURE_COLS].astype("float64").reset_index(drop=True)

    report = Report(metrics=[DataDriftPreset(drift_share=DATA_DRIFT_THRESHOLD)])
    report.run(reference_data=ref, current_data=curr)

    # Saving HTML report to output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info(f"Drift report saved to : {output_path.name}")

    # Creating summary dict
    summary = parse_report_summary(report.as_dict())

    logger.info(
        f"Drift Share : {summary['drift_share']:.2f} | "
        f"Features Drifted : {summary['n_drifted']} / {summary['n_features']}"
    )
    return summary


def run_drift_detection(
    year_start: int = DRIFT_YEAR,
    year_end: int = DRIFT_YEAR,
    sample_size: int = SAMPLE_SIZE,
    output_path: Path = None
) -> dict: 
    # Full drift detection pipeline for 1 year window

    if output_path is None:
        output_path = DRIFT_REPORT_PATH

    from src.monitoring.logger import log_drift_detected, log_drift_report_generated

    reference_period = f"{V1_TRAIN_START}-{V1_TRAIN_END}"
    current_period = (
        f"{year_start}-{year_end}"
        if year_start != year_end
        else str(year_start)
    )

    logger.info(
        f"Drift Detection | "
        f"Reference : {reference_period} | Current : {current_period}"
    )

    ref_df = load_reference_data(sample_size=sample_size)
    curr_df = load_current_data(
        year_start=year_start,
        year_end=year_end,
        sample_size=sample_size
    )

    summary = generate_drift_report(
        reference_df=ref_df,
        current_df=curr_df,
        reference_period=reference_period,
        current_period=current_period,
        output_path=output_path
    )

    log_drift_report_generated(
        output_path=str(output_path),
        reference_period=reference_period,
        current_period=current_period,
        n_reference_rows=len(ref_df),
        n_current_rows=len(curr_df)
    )

    if summary["n_drifted"] > 0:
        log_drift_detected(
            drift_share= summary["drift_share"],
            n_drifted= summary["n_drifted"],
            n_features= summary["n_features"],
            drifted_features= summary["drifted_features"],
            reference_period=reference_period,
            current_period=current_period
        )

    return summary

if __name__ == "__main__":
    run_drift_detection()