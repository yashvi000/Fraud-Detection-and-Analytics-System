import pytest
import pandas as pd
import numpy as np
import sys
import yaml
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_PATH = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_PATH / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


FEATURE_COLS = config["feature_cols"]

def mock_build_features():
    return pd.DataFrame([np.zeros(38)], columns=FEATURE_COLS)


class TestFullPipeline:
    # Tests POST /predict end-to-end

    def test_predict_stores_in_predictions_table(self, integration_app, critical_transaction):
        # Checks if predictions from POST /predict are stored correctly

        with patch("api.routes.predict.build_features") as mock_build, \
            patch("api.routes.predict.insert_prediction") as mock_insert, \
            patch("api.routes.predict.insert_scored_transaction"):

            mock_build.return_value = mock_build_features()

            response = integration_app.post("/predict", json=critical_transaction)
            assert response.status_code == 200

            stored = mock_insert.call_args[0][0]
            assert stored["transaction_id"] == "INT_TEST_CRITICAL_001"
            assert stored["is_alert"] == True
            assert stored["fraud_probability"] == round(0.97, 6)
            assert stored["risk_tier"] == "critical"
            assert stored["inference_latency"] >= 0

    
    def test_critical_transaction_triggers_shap(self, integration_app, critical_transaction):
        # Critical transactions should trigger SHAP Explaination

        with patch("api.routes.predict.build_features") as mock_build, \
            patch("api.routes.predict.insert_prediction"), \
            patch("api.routes.predict.insert_scored_transaction"), \
            patch("api.routes.predict.explain_transaction") as mock_shap:

            mock_build.return_value = mock_build_features()

            mock_shap.return_value = {"card_is_new_merchant": 0.42, "is_online": 0.31}

            response = integration_app.post("/predict", json=critical_transaction)
            data = response.json()

            assert response.status_code == 200
            assert data["shap_explainability"] is not None
            assert mock_shap.called

    
    def test_minimal_transaction_does_not_triggers_shap(self, mock_app, sample_txn):
        # Minimal transactions should not trigger SHAP Explaination

        with patch("api.routes.predict.build_features") as mock_build, \
            patch("api.routes.predict.insert_prediction"), \
            patch("api.routes.predict.insert_scored_transaction"), \
            patch("api.routes.predict.explain_transaction") as mock_shap:

            mock_build.return_value = mock_build_features()

            response = mock_app.post("/predict", json=sample_txn)
            data = response.json()

            assert response.status_code == 200
            assert data["shap_explainability"] is None
            assert not mock_shap.called


    def test_user_txn_stored_with_source_as_user(self, integration_app, critical_transaction):
        # Checks if txns with transaction_id 'USER_...' are stored in scored_transactions with source = user

        txn = {**critical_transaction, "transaction_id": "USER_INT_TEST_001"}

        with patch("api.routes.predict.build_features") as mock_build, \
            patch("api.routes.predict.insert_prediction"), \
            patch("api.routes.predict.insert_scored_transaction") as mock_scored:

            mock_build.return_value = mock_build_features()

            response = integration_app.post("/predict", json=txn)
            assert response.status_code == 200

            assert mock_scored.called
            stored = mock_scored.call_args[0][0]
            assert stored["source"] == "user"
            assert stored["transaction_id"] == "USER_INT_TEST_001"


    def test_demo_txn_stored_with_source_as_demo(self, integration_app, critical_transaction):
        # Checks if txns with transaction_id 'DEMO_...' are stored in scored_transactions with source = demo

        txn = {**critical_transaction, "transaction_id": "DEMO_INT_TEST_001"}

        with patch("api.routes.predict.build_features") as mock_build, \
            patch("api.routes.predict.insert_prediction"), \
            patch("api.routes.predict.insert_scored_transaction") as mock_scored:

            mock_build.return_value = mock_build_features()

            response = integration_app.post("/predict", json=txn)
            assert response.status_code == 200

            assert mock_scored.called
            stored = mock_scored.call_args[0][0]
            assert stored["source"] == "demo"
            assert stored["transaction_id"] == "DEMO_INT_TEST_001"


    def test_real_txn_not_stored_in_scored_transactions(self, integration_app, critical_transaction):
        # Checks if other txns are not stored in scored_transactions table

        with patch("api.routes.predict.build_features") as mock_build, \
            patch("api.routes.predict.insert_prediction"), \
            patch("api.routes.predict.insert_scored_transaction") as mock_scored:

            mock_build.return_value = mock_build_features()

            response = integration_app.post("/predict", json=critical_transaction)
            assert response.status_code == 200

            assert not mock_scored.called
    

    def test_scored_transactions_has_all_fields(self, integration_app, critical_transaction):
        # scored_transactions table should have all fields for Live Alerts table

        txn = {**critical_transaction, "transaction_id": "USER_INT_TEST_002"}

        with patch("api.routes.predict.build_features") as mock_build, \
            patch("api.routes.predict.insert_prediction"), \
            patch("api.routes.predict.insert_scored_transaction") as mock_scored:

            mock_build.return_value = mock_build_features()
            integration_app.post("/predict", json=txn)

            stored = mock_scored.call_args[0][0]

            required = [
                "transaction_id", "user_id", "card", "timestamp", "amount", 
                "merchant_city", "merchant_state", "use_chip", "mcc", "source"
            ]

            for field in required:
                assert field in stored, f"Missing in scored_transactions : {field}"

    
class TestRegressionConsistency:

    def test_same_input_always_gives_same_output(self, integration_app, critical_transaction):
        # Same txn should return exactly same results
        # Finds any config or logic changes

        with patch("api.routes.predict.build_features") as mock_build, \
            patch("api.routes.predict.insert_prediction"), \
            patch("api.routes.predict.insert_scored_transaction"):

            mock_build.return_value = mock_build_features()

            r1 = integration_app.post("/predict", json=critical_transaction).json()
            r2 = integration_app.post("/predict", json=critical_transaction).json()

            assert r1["fraud_probability"] == r2["fraud_probability"]
            assert r1["risk_tier"] == r2["risk_tier"]
            assert r1["is_alert"] == r2["is_alert"]
            assert r1["expected_exposure"] == r2["expected_exposure"]

    
    def test_threshold_determines_alert_and_tier(self, integration_app, critical_transaction):
        # Finds any threshold or risk tier config changes

        with patch("api.routes.predict.build_features") as mock_build, \
            patch("api.routes.predict.insert_prediction"), \
            patch("api.routes.predict.insert_scored_transaction"):

            mock_build.return_value = mock_build_features()
            data = integration_app.post("/predict", json=critical_transaction).json()

            assert data["is_alert"] == True
            assert data["risk_tier"] == "critical"

    
    def test_exposure_formula_is_same(self, integration_app, critical_transaction):
        # Finds any changes to exposure formula (amount * fraud_prob * severity_weight)
        # For critical_transaction, expected exposure should be 500 * 0.97 * 1.5 = 727.50

        with patch("api.routes.predict.build_features") as mock_build, \
            patch("api.routes.predict.insert_prediction"), \
            patch("api.routes.predict.insert_scored_transaction"):

            mock_build.return_value = mock_build_features()
            data = integration_app.post("/predict", json=critical_transaction).json()

            expected = round(500 * 0.97 * 1.5, 2)
            assert abs(data["expected_exposure"] - expected) < 0.01, \
                f"Exposure formula changed : expected {expected}, got {data['expected_exposure']}"
            
    
    def test_latency_always_positive(self, integration_app, critical_transaction):
        # Catches timing logic removal

        with patch("api.routes.predict.build_features") as mock_build, \
            patch("api.routes.predict.insert_prediction"), \
            patch("api.routes.predict.insert_scored_transaction"):

            mock_build.return_value = mock_build_features()
            data = integration_app.post("/predict", json=critical_transaction).json()
            assert data["inference_latency"] >= 0

    
    def test_transaction_id_preserved_in_response(self, integration_app, sample_txn):
        # Response transaction_id should match request transaction_id

        with patch("api.routes.predict.build_features") as mock_build, \
            patch("api.routes.predict.insert_prediction"), \
            patch("api.routes.predict.insert_scored_transaction"):

            mock_build.return_value = mock_build_features()
            data = integration_app.post("/predict", json=sample_txn).json()
            
            assert data["transaction_id"] == sample_txn["transaction_id"]