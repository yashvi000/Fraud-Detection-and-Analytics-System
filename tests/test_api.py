import pytest
import pandas as pd
import numpy as np
import sys
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Fixtures
@pytest.fixture
def mock_app():
    # Creating FastAPI test client

    from api.main import app
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])

    PROJECT_PATH = Path(__file__).resolve().parents[1]
    CONFIG_PATH = PROJECT_PATH / "config" / "config.yaml"

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    app.state.model = mock_model
    app.state.mcc_freq = {}
    app.state.cold_start = {}
    app.state.config = config
    app.state.threshold = config["model"]["threshold"]
    app.state.feature_cols = config["feature_cols"]

    return TestClient(app)


@pytest.fixture
def sample_txn():
    # Sample transaction payload for POST /predict

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


class TestHealth:

    def test_health_returns_200_and_ok(self, mock_app):
        response = mock_app.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAlerts:

    def test_alerts_returns_200(self, mock_app):
        response = mock_app.get("/alerts")
        assert response.status_code == 200

    def test_alerts_returns_correct_keys(self, mock_app):
        response = mock_app.get("/alerts")
        data = response.json()
        assert "alerts" in data
        assert "total" in data
    
    def test_alerts_total_matches_alerts_length(self, mock_app):
        response = mock_app.get("/alerts")
        data = response.json()
        assert data["total"] == len(data["alerts"])

    def test_alerts_limits_correctly(self, mock_app):
        response = mock_app.get("/alerts?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] <= 5

    def test_alerts_limit_too_small_returns_422(self, mock_app):
        # Limit less than 1 gets status 422
        response = mock_app.get("/alerts?limit=0")
        assert response.status_code == 422
    
    def test_alerts_limit_too_large_returns_422(self, mock_app):
        # Limit more than 500 gets status 422
        response = mock_app.get("/alerts?limit=501")
        assert response.status_code == 422


class TestMetrics:

    def test_metrics_returns_200(self, mock_app):
        response = mock_app.get("/metrics")
        assert response.status_code == 200

    def test_metrics_returns_correct_keys(self, mock_app):
        response = mock_app.get("/metrics")
        data = response.json()
        
        expected = [
            "total_predictions",
            "total_alerts",
            "alert_rate",
            "avg_fraud_prob",
            "avg_exposure",
            "avg_latency_ms",
            "risk_tiers",
            "alert_over_time",
        ]

        for key in expected:
            assert key in data, f"Missing key : {key}"
    
    def test_metrics_datatype(self, mock_app):
        response = mock_app.get("/metrics")
        data = response.json()

        assert isinstance(data["total_predictions"], int)
        assert isinstance(data["total_alerts"], int)
        assert isinstance(data["alert_rate"], float)
        assert isinstance(data["avg_fraud_prob"], float)
        assert isinstance(data["avg_exposure"], float)
        assert isinstance(data["avg_latency_ms"], float)
        assert isinstance(data["risk_tiers"], dict)
        assert isinstance(data["alert_over_time"], list)

    def test_metrics_risk_tiers_keys_are_valid(self, mock_app):
        response = mock_app.get("/metrics")
        data = response.json()

        valid_tiers = {
            "minimal",
            "low",
            "medium",
            "high",
            "critical"
        }

        for tier in data["risk_tiers"].keys():
            assert tier in valid_tiers, f"Invalid risk tier in metrics : {tier}"


class TestPredict:

    def test_predict_returns_200(self, mock_app, sample_txn):

        with patch("api.routes.predict.build_features") as mock_build, \
             patch("api.routes.predict.insert_prediction"):
            
            mock_build.return_value = pd.DataFrame(
                [np.zeros(38)],
                columns= mock_app.app.state.feature_cols
            )

            response = mock_app.post("/predict", json=sample_txn)
            assert response.status_code == 200

    def test_predict_returns_correct_keys(self, mock_app, sample_txn):

        with patch("api.routes.predict.build_features") as mock_build, \
             patch("api.routes.predict.insert_prediction"):
            
            mock_build.return_value = pd.DataFrame(
                [np.zeros(38)],
                columns= mock_app.app.state.feature_cols
            )

            response = mock_app.post("/predict", json=sample_txn)
            data = response.json()

            expected = [
                "transaction_id",
                "fraud_probability",
                "risk_tier",
                "expected_exposure",
                "is_alert",
                "inference_latency",
            ]

            for key in expected:
                assert key in data, f"Missing key : {key}"
    
    def test_predict_missing_field_returns_422(self, mock_app):
        
        incomplete = {
            "transaction_id" : "TEST_TXN_002",
            "user_id" : 0,
            "timestamp" : "2013-08-11 16:40:00",
            "amount" : 325.00,
        }

        response = mock_app.post("/predict", json=incomplete)
        assert response.status_code == 422
    
    def test_predict_wrong_type_returns_422(self, mock_app, sample_txn):
        # String instead of integer should return 422

        bad_payload = sample_txn.copy()
        bad_payload["amount"] = "143.oo"  # string
        
        response = mock_app.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_transaction_id_matches(self, mock_app, sample_txn):
        # Request transaction_id should match response transaction_id

        with patch("api.routes.predict.build_features") as mock_build, \
             patch("api.routes.predict.insert_prediction"):
            
            mock_build.return_value = pd.DataFrame(
                [np.zeros(38)],
                columns= mock_app.app.state.feature_cols
            )

            response = mock_app.post("/predict", json=sample_txn)
            data = response.json()
            assert data["transaction_id"] == sample_txn["transaction_id"]