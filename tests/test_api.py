import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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