import streamlit as st
import altair as alt
import requests
import pandas as pd
import yaml
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.db import engine, get_batch_scored_transactions
from sqlalchemy import text

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Loading paths and config
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

SHAP_PLOT_PATH = PROJECT_ROOT / config["artifacts"]["shap_global_plot_path"]
DRIFT_REPORT_PATH = PROJECT_ROOT / config["monitoring"]["drift_report_path"]
REFRESH_SECONDS = config["dashboard"]["refresh_seconds"]

V1_TRAIN_START = config["splits"]["v1_train_start"]
V1_TRAIN_END = config["splits"]["v1_train_end"]
V1_VAL_START = config["splits"]["v1_val_start"]
V1_VAL_END = config["splits"]["v1_val_end"]
V1_TEST_START = config["splits"]["v1_test_start"]
V1_TEST_END = config["splits"]["v1_test_end"]

THRESHOLD = config["model"]["threshold"]
LATENCY_TARGET_MS = config["monitoring"]["latency_target_ms"]
ALGORITHM = config["model"]["lgbm"]["algorithm"]
ERROR_OPTIONS = config["dashboard"]["error_options"]

# Loading from .env
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"


# Getting API responses
def fetch_metrics() -> dict:
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        return response.json()
    except Exception:
        return {}
    
def fetch_alerts(limit: int = 50) -> list:
    try:
        response = requests.get(f"{API_URL}/alerts?limit={limit}", timeout=5)
        return response.json().get("alerts", [])
    except Exception:
        return []


# Loading states and cities from PostgreSQL
@st.cache_data(ttl=3600)
def load_unique_states() -> list:
    
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT merchant_state
            FROM transactions
            WHERE merchant_state != 'ONLINE'
            ORDER BY merchant_state ASC
        """)).fetchall()

    return [row[0] for row in rows]

@st.cache_data(ttl=3600)
def load_cities_by_states() -> dict:
    
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT merchant_state, merchant_city
            FROM transactions
            WHERE merchant_state != 'ONLINE' AND merchant_city != 'ONLINE'
            ORDER BY merchant_state, merchant_city ASC
        """)).fetchall()
    
    state_cities = {}
    for row in rows:
        state = row[0]
        city = row[1]

        if state not in state_cities:
            state_cities[state] = []
        state_cities[state].append(city)

    return state_cities
    

# Page config
st.set_page_config(
    page_title= "Fraud Detection Dashboard",
    layout= "wide"
)

# Sidebar
st.sidebar.title("Fraud Detection")
st.sidebar.caption(f"{MODEL_VERSION} Model | {V1_TRAIN_START}-{V1_TEST_END}")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Live Alerts", "Model Insights", "Score Transactions"],
    index=0
)
st.sidebar.divider()


# Auto-Refreshing
auto_refresh = st.sidebar.toggle(
    f"Auto-Refresh ({REFRESH_SECONDS}s)",
    value=True
)

if auto_refresh:
    st.sidebar.caption(
        f"Last Refresh : {pd.Timestamp.now().strftime('%H:%M:%S')}"
    )


# Loading data
metrics = fetch_metrics()
alerts = fetch_alerts(limit=50)


# Page 1- Overview

if page == "Overview":
    st.title("Fraud Detection Overview")
    st.caption(f"Real-time fraud analytics from {MODEL_VERSION} model predictions")
    st.divider()

    if not metrics:
        st.error("Cannot connect to API. Make sure FastAPI is running.")
        st.stop()
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label= "Total Predictions",
        value= f"{metrics.get('total_predictions', 0):,}"
    )

    col2.metric(
        label= "Total Alerts",
        value= f"{metrics.get('total_alerts', 0):,}"
    )

    col3.metric(
        label= "Alert Rate",
        value= f"{metrics.get('alert_rate', 0):.4f}"
    )

    col4.metric(
        label= "Average Exposure",
        value= f"${metrics.get('avg_exposure', 0):.2f}"
    )

    st.divider()

    col_time, col_risk = st.columns(2)

    # Alert Rate Over Time (Line Chart)
    with col_time:
        st.subheader("Alert Rate Over Time")
        alert_over_time = metrics.get("alert_over_time", [])
        
        if alert_over_time:
            df_time = pd.DataFrame(alert_over_time)
            df_time["month"] = pd.to_datetime(df_time["month"])
            df_time["alert_rate"] = df_time["alerts"] / df_time["total"].replace(0, 1) * 100
            
            chart = alt.Chart(df_time).mark_line(point=True).encode(
                x= alt.X("month:T", axis=alt.Axis(format="%d %b'%y", labelAngle=0)),
                y= alt.Y("alert_rate:Q", title= "Alert Rate (%)")
            )
            st.altair_chart(chart, use_container_width=True)
        
        else:
            st.info("No data available")
    
    # Risk Tier Distribution (Bar Graph)
    with col_risk:
        st.subheader("Risk Tier Distribution")
        risk_tiers = metrics.get("risk_tiers", [])

        if risk_tiers:
            tier_order = ["minimal", "low", "medium", "high", "critical"]
            df_tiers = pd.DataFrame([
                {"tier" : k, "count" : v}
                for k, v in risk_tiers.items()
            ])

            df_tiers["tier"] = pd.Categorical(
                df_tiers["tier"], categories=tier_order, ordered=True
            )

            df_tiers = df_tiers.sort_values("tier").set_index("tier")
            st.bar_chart(df_tiers["count"], use_container_width=True)
        
        else:
            st.info("No data available")

    st.divider()

    # Alert Volume Over Time (Line Chart)
    st.subheader("Alert Volume Over Time")

    if alert_over_time:
        df_alert_vol = pd.DataFrame(alert_over_time)
        df_alert_vol["month"] = pd.to_datetime(df_alert_vol["month"])
        
        chart = alt.Chart(df_alert_vol).mark_line(point=True).encode(
            x= alt.X("month:T", axis=alt.Axis(format="%d %b'%y", labelAngle=0)),
            y= alt.Y("alerts:Q", title= "Alert Count")
        )
        st.altair_chart(chart, use_container_width=True)
    
    else:
        st.info("No data available")


# Page 2- Live Alerts

elif page == "Live Alerts":
    st.title("Live Fraud Alert Feed")
    st.caption(f"Most recent fraud alerts (refreshes every {REFRESH_SECONDS}s)")
    st.divider()

    limit = st.slider(
        "Number of alerts to display",
        min_value=10, max_value=500, value=50, step=10
    )

    alerts = fetch_alerts(limit=limit)

    if not alerts:
        st.info("No fraud alerts found in predictions table")
    
    else:
        df_alerts = pd.DataFrame(alerts)

        user_or_demo_ids = [
            row["transaction_id"]
            for _, row in df_alerts.iterrows()
            if str(row.get("transaction_id", "")).startswith(("USER_", "DEMO_"))
        ]

        scored_store = get_batch_scored_transactions(user_or_demo_ids)

        def fill_txn_columns(row):
            txn_id = row.get("transaction_id", "")

            if not txn_id.startswith(("USER_", "DEMO_")):
                return row

            store = scored_store.get(txn_id)
            if store:
                row["user_id"] = store["user_id"]
                row["card"] = store["card"]
                row["amount"] = store["amount"]
                row["merchant_city"] = store["merchant_city"]
                row["merchant_state"] = store["merchant_state"]
                row["use_chip"] = store["use_chip"]
                row["txn_timestamp"] = str(store["timestamp"])
            
            return row

        df_alerts = df_alerts.apply(fill_txn_columns, axis=1)
        

        if "fraud_probability" in df_alerts.columns:
            df_alerts["fraud_probability"] = df_alerts["fraud_probability"].round(4)
        
        if "expected_exposure" in df_alerts.columns:
            df_alerts["expected_exposure"] = (
                df_alerts["expected_exposure"]
                .apply(lambda x: f"${x:,.2f}" if x else "$0.00")
            )

        if "amount" in df_alerts.columns:
            df_alerts["amount"] = (
                df_alerts["amount"]
                .apply(lambda x: f"${x:,.2f}" if x is not None and x != "N/A" else "N/A")
            )
        
        
        display_cols = [
            c for c in [
                "transaction_id", "fraud_probability", "risk_tier",
                "expected_exposure", "amount", "user_id", "card", 
                "merchant_city", "merchant_state", "use_chip", 
                "txn_timestamp", "predicted_at", "model_version"
            ] if c in df_alerts.columns
        ]

        st.dataframe(
            df_alerts[display_cols],
            use_container_width=True,
            height=600
        )

        st.caption(f"Showing {len(df_alerts):,} most recent alerts")


# Page 3- Model Insights

elif page == "Model Insights":
    st.title("Model Insights")
    st.divider()

    # Model Information
    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Algorithm", ALGORITHM)
    col2.metric("Decision Threshold", f"{THRESHOLD}")
    col3.metric("Features", f"{len(config['feature_cols'])}")

    col4, col5, col6, col7 = st.columns(4)
    col4.metric("Training Period", f"{V1_TRAIN_START} - {V1_TRAIN_END}")
    col5.metric("Validation Period", f"{V1_VAL_START} - {V1_VAL_END}")
    col6.metric("Testing Period", f"{V1_TEST_START} - {V1_TEST_END}")
    col7.metric("Model Version", MODEL_VERSION)

    st.divider()


    # SHAP Global Importance
    st.subheader("SHAP Global Feature Importance")
    st.caption("Computed on stratified training sample")

    if SHAP_PLOT_PATH.exists():
        st.image(
            str(SHAP_PLOT_PATH),
            caption="Mean |SHAP| value across training sample",
            use_container_width=True
        )
    
    else:
        st.warning(f"SHAP plot not found at {SHAP_PLOT_PATH}")
    st.divider()


    # Inference Performance
    st.subheader("Inference Performance")

    if metrics:
        col1, col2 = st.columns(2)
        
        col1.metric(
            "Average Latency",
            f"{metrics.get('avg_latency_ms', 0):.1f} ms",
            delta= f"Target : {LATENCY_TARGET_MS} ms"
        )

        col2.metric(
            "Average Fraud Probability",
            f"{metrics.get('avg_fraud_prob', 0):.4f}"
        )


# Page 4- Score Transactions

elif page == "Score Transactions":
    st.title("Score a Transaction")
    st.caption("Submit a transaction to get a real-time fraud score")
    st.divider()

    # transaction_id for USER
    transaction_id = f"USER_TXN_{uuid.uuid4().hex[:8].upper()}"
    st.caption(f"Transaction ID : **{transaction_id}**")

    # Loading states and cities
    STATES = load_unique_states()
    state_cities = load_cities_by_states()

    pre_col1, pre_col2, pre_col3 = st.columns(3)

    with pre_col1:

        timestamp = st.text_input(
            "Timestamp (YYYY-MM-DD HH:MM:SS)",
            value= "2013-06-15 14:23:00",
            help= "Use a date within 2013-2014 for V1 model accuracy"
        )


    try:
        ts_year = int(timestamp[:4])
    except Exception:
        ts_year = 2013
    
    with pre_col2:
        if ts_year >= 2015:
            use_chip_options = ["swipe", "chip", "online"]
        else:
            use_chip_options = ["swipe", "online"]
        
        use_chip = st.selectbox(
            "Transaction Method",
            options= use_chip_options,
            help= "swipe : physical card swipe | online : without physical card | chip : EMV chip (2015+)"
        )

        if ts_year < 2015:
            st.caption("Chip cards were not available before 2015")
    

    with pre_col3:
        if use_chip == "online":
            merchant_state = "ONLINE"
            
            st.selectbox(
                "Merchant State / Country",
                options=["ONLINE"],
                disabled=True,
                help="Online transactions have no physical location"
            )
        
        else:
            merchant_state = st.selectbox(
                "Merchant State / Country",
                options=STATES,
                index=0,
                help="Includes US States & International locations from dataset"
            )

            valid_cities = state_cities.get(merchant_state, [])

            if valid_cities:
                display_cities = valid_cities[:20]
                more = len(valid_cities) - 20
                city_hint = ", ".join(display_cities)

                if more > 0:
                    city_hint += f" ... (+{more} more)"
                st.info(f"Cities in {merchant_state} : {city_hint}")
            

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            user_id = st.number_input(
                "User ID (0 - 1999)",
                min_value=0, max_value=1999,
                value=0, step=1,
                help="IBM dataset users range from 0 to 1999"
            )

            card = st.number_input(
                "Card (0 - 8)",
                min_value=0, max_value=8,
                value=0, step=1,
                help="Each user has upto 9 cards numbered from 0 to 8"
            )

            amount = st.number_input(
                "Amount ($)",
                value=100.00,
                step=0.01,
                help= "For refund transaction, enter negative amount"
            )


        with col2:
            merchant_name = st.number_input(
                "Merchant ID",
                value= 9999999999,
                help= "Numeric merchant identifier from IBM dataset"
            )

            mcc = st.number_input(
                "MCC (Merchant Category Code)",
                min_value=0, value=5912,
                step=1,
                help= "4-digit code indentifying merchant category"
            )

            if use_chip == "online":

                st.text_input(
                    "Merchant City",
                    value="ONLINE",
                    disabled=True,
                    help="Online transactions have no physical location"
                )
                merchant_city = "ONLINE"
            
            else:

                merchant_city = st.text_input(
                    "Merchant City",
                    value=valid_cities[0] if valid_cities else "",
                    help="Type a city from the reference above"
                )
            
            selected_errors = st.selectbox(
                "Transaction Errors",
                options=ERROR_OPTIONS,
                index=0,
                help= "Select any errors with this transaction"
            )
        
        submitted = st.form_submit_button(
            "Score Transaction",
            use_container_width= True
        )
    
    if submitted:
        payload = {
            "transaction_id" : transaction_id,
            "user_id" : int(user_id),
            "card" : int(card),
            "timestamp" : timestamp,
            "amount" : float(amount),
            "use_chip" : use_chip,
            "merchant_name" : int(merchant_name),
            "merchant_city" : merchant_city,
            "merchant_state" : merchant_state,
            "mcc" : int(mcc),
            "errors" : selected_errors,
        }

        with st.spinner("Scoring your transaction ..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    st.divider()

                    st.subheader("Fraud Score Result")
                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric("Fraud Probability", f"{result['fraud_probability']:.6f}")
                    col2.metric("Risk Tier", result["risk_tier"].upper())
                    col3.metric("Expected Exposure", f"${result['expected_exposure']:,.2f}")
                    col4.metric("Is Alert", "YES" if result["is_alert"] else "NO")

                    st.metric("Inference Latency", f"{result['inference_latency']} ms")

                    if result.get("shap_explainability"):
                        st.divider()
                        st.subheader("SHAP Explaination - Top 10 Features")
                        st.caption(
                            "Red : pushes towards fraud | Green : pushes towards not fraud"
                        )

                        shap_df = pd.DataFrame([
                            {"feature": k, "shap_value": round(v, 5)}
                            for k, v in result["shap_explainability"].items()
                        ]).sort_values(
                            "shap_value", key=abs, ascending=False
                        ).head(10)

                        chart = alt.Chart(shap_df).mark_bar().encode(
                            x=alt.X("shap_value:Q", title="SHAP Value"),
                            y=alt.Y("feature:N", sort="-x", title=""),
                            color=alt.condition(
                                alt.datum.shap_value > 0,
                                alt.value("red"),
                                alt.value("green")
                            )
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
                    
                    else:
                        st.info(
                            "SHAP Explaination (only for high or critical risk transactions)"
                        )
                
                else:
                    st.error(f"API Error {response.status_code} : {response.text}")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Cannot connect to API : {e}")

# Auto-Refresh
if auto_refresh:
    import time
    time.sleep(REFRESH_SECONDS)
    st.rerun()