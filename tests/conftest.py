import pytest
import pandas as pd
import numpy as np
import sys
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from api.main import app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_PATH = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_PATH / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


@pytest.fixture
def sample_df():
    # For testing feature pipeline

    return pd.DataFrame({
        "user_id" : [0, 0, 0, 0, 1, 1, 1, 1],
        "card" : [0, 0, 1, 1, 0, 0, 0, 0],
        "timestamp" : pd.to_datetime([
            "2005-01-01 10:00",
            "2005-01-01 10:30",
            "2005-01-02 01:00",
            "2011-03-25 10:00",
            "2013-01-15 09:00",
            "2013-02-01 09:05",
            "2013-03-14 09:10",
            "2013-04-05 09:30"
        ]),
        "amount" : [100.0, 200.0, 50.0, 300.0, 80.0, 90.0, 40.0, 30.0],
        "is_refund" : [0, 0, 0, 0, 0, 0, 0, 0],
        "mcc" : [5411, 5411, 5912, 5411, 5411, 5912, 7777, 5411],
        "use_chip" : ["swipe", "online", "swipe", "swipe", "swipe", "online", "swipe", "swipe"],
        "merchant_name" : ["1001", "1002", "1001", "1001", "2001", "2001", "2002", "2001"],
        "merchant_state" : ["CA", "ONLINE", "CA", "NY", "TX", "ONLINE", "TX", "TX"],
        "merchant_city" : ["LA", "ONLINE", "LA", "NYC", "Austin", "ONLINE", "Austin", "Austin"],
        "is_fraud" : [0, 0, 0, 0, 0, 0, 1, 0]
    })


@pytest.fixture
def sample_txn():
    # Sample transaction payload for POST /predict
    # For testing API and integration tests

    return {
        "transaction_id" : "TEST_TXN_001",
        "user_id" : 0,
        "card" : 0,
        "timestamp" : "2013-04-01 16:40:00",
        "amount" : 143.00,
        "use_chip" : "swipe",
        "merchant_name" : 999999999,
        "merchant_city" : "La Verne",
        "merchant_state" : "CA",
        "mcc" : 5912,
        "errors" : "None",
    }


@pytest.fixture
def mock_app():
    # Creating FastAPI test client with minimal fraud score
    # For API testing and Integration Tests

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])

    app.state.model = mock_model
    app.state.mcc_freq = {}
    app.state.cold_start = {}
    app.state.config = config
    app.state.threshold = config["model"]["threshold"]
    app.state.feature_cols = config["feature_cols"]

    return TestClient(app)


@pytest.fixture
def integration_app():
    # Creating FastAPI test client with critical fraud score
    # For Integration Tests

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.03, 0.97]])

    app.state.model = mock_model
    app.state.mcc_freq = {}
    app.state.cold_start = {}
    app.state.config = config
    app.state.threshold = config["model"]["threshold"]
    app.state.feature_cols = config["feature_cols"]

    return TestClient(app)


@pytest.fixture
def critical_transaction():
    # Critical transaction payload for integration tests
    
    return {
        "transaction_id" : "INT_TEST_CRITICAL_001",
        "user_id" : 0,
        "card" : 0,
        "timestamp" : "2013-09-01 11:40:00",
        "amount" : 500.00,
        "use_chip" : "online",
        "merchant_name" : 999999999,
        "merchant_city" : "NYC",
        "merchant_state" : "NY",
        "mcc" : 5912,
        "errors" : "None"
    }