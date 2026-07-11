import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.model_utils import evaluate_model

# Stub Model
class StubModel:
    def __init__(self, proba: np.ndarray):
        self.proba = np.array(proba)

    def predict_proba(self, x):
        return np.column_stack([1 - self.proba, self.proba])
    

# Fixtures
@pytest.fixture
def binary_labels():
    # Fixed labels- 10 fraud, 90 non-fraud
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    return pd.Series(y)

@pytest.fixture
def perfect_model(binary_labels):
    # Fraud prob = 1.00, Non-Fraud prob = 0.00
    # Recall = 1.00, fpr = 0.00, precision = 1.00 at all thresholds
    proba = np.where(binary_labels == 1, 1.0, 0.0)
    return StubModel(proba)

@pytest.fixture
def random_model():
    # Random probabilities
    np.random.seed(42)
    return StubModel(np.random.uniform(0, 1, 100))

@pytest.fixture
def worst_model(binary_labels):
    # Fraud prob = 0.00, Non-Fraud prob = 1.00
    # Recall = 0.00, fpr = 1.00, precision = 0.00 at all thresholds
    proba = np.where(binary_labels == 1, 0.0, 1.0)
    return StubModel(proba)

@pytest.fixture
def sample_x():
    # Dummy feature matrix with shape = (100, 5)
    return pd.DataFrame(np.zeros((100, 5)))


# Tests
class TestEvaluateModel:

    def test_all_expected_keys_present(self, perfect_model, sample_x, binary_labels):
        
        _, metrics = evaluate_model(perfect_model, sample_x, binary_labels, threshold=0.5)
        
        expected = [
            "pr_auc", "roc_auc", "f1", "recall", "precision", "fpr", 
            "recall_at_0.1_fpr", "recall_at_1_fpr", "precision_top_1k", 
            "precision_top_5k", "precision_top_10k"
        ]
        
        for key in expected:
            assert key in metrics, f"Missing Key : {key}"
    

    def test_all_metric_values_are_floats(self, perfect_model, sample_x, binary_labels):

        _, metrics = evaluate_model(perfect_model, sample_x, binary_labels, threshold=0.5)
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not float : {type(value)}"


    def test_perfect_model_metrics(self, perfect_model, sample_x, binary_labels):
        # Recall = 1.0, fpr = 0.0, pr_auc = 1.0, roc_auc = 1.0, precision = 1.0
        
        _, metrics = evaluate_model(perfect_model, sample_x, binary_labels, threshold=0.5)
        assert metrics["recall"] == 1.0
        assert metrics["pr_auc"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["fpr"] == 0.0
        assert metrics["precision"] == 1.0


    def test_worst_model_metrics(self, worst_model, sample_x, binary_labels):
        # Recall = 0.0, fpr = 1.0, precision = 0.0
        
        _, metrics = evaluate_model(worst_model, sample_x, binary_labels, threshold=0.5)
        assert metrics["recall"] == 0.0
        assert metrics["fpr"] == 1.0
        assert metrics["precision"] == 0.0


    def test_threshold_above_max_gives_zero_recall_and_fpr(self, perfect_model, sample_x, binary_labels):
        # At threshold > 1, nothing is flagged so recall = 0.0, fpr = 0.0

        _, metrics = evaluate_model(perfect_model, sample_x, binary_labels, threshold=1.01)
        assert metrics["recall"] == 0.0
        assert metrics["fpr"] == 0.0
    

    def test_threshold_0_gives_full_recall_and_fpr(self, perfect_model, sample_x, binary_labels):
        # At threshold = 0.0, everything is flagged so recall = 1.0, fpr = 1.0

        _, metrics = evaluate_model(perfect_model, sample_x, binary_labels, threshold=0.0)
        assert metrics["recall"] == 1.0
        assert metrics["fpr"] == 1.0


    def test_higher_threshold_lowers_recall_and_fpr(self, random_model, sample_x, binary_labels):

        _, low = evaluate_model(random_model, sample_x, binary_labels, threshold=0.3)
        _, high = evaluate_model(random_model, sample_x, binary_labels, threshold=0.7)

        assert high["recall"] <= low["recall"]
        assert high["fpr"] <= low["fpr"]


    def test_precision_at_k_returns_0_when_k_exceeds_dataset(self, random_model, sample_x, binary_labels):
        # Dataset has 100 rows, precision at top 1k, 5k, and 10k should be 0.0

        _, metrics = evaluate_model(random_model, sample_x, binary_labels, threshold=0.5)
        assert metrics["precision_top_1k"] == 0.0
        assert metrics["precision_top_5k"] == 0.0
        assert metrics["precision_top_10k"] == 0.0


    def test_recall_at_fpr_returns_float(self, random_model, sample_x, binary_labels):
        _, metrics = evaluate_model(random_model, sample_x, binary_labels, threshold=0.5)
        
        assert isinstance(metrics["recall_at_0.1_fpr"], float)
        assert isinstance(metrics["recall_at_1_fpr"], float)


    def test_prob_length_and_range(self, perfect_model, sample_x, binary_labels):
        prob, _ = evaluate_model(perfect_model, sample_x, binary_labels, threshold=0.5)
        
        assert len(prob) == len(binary_labels)
        assert (prob >= 0.0).all() and (prob <= 1.0).all()
    