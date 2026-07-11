import pytest
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk.scoring import compute_risk_tier, compute_expected_exposure


class TestComputeRiskTier:

    def test_valid_tier_values(self):
        # All probabilities should be mapped to a tier

        valid_tiers = {"minimal", "low", "medium", "high", "critical"}
        for prob in [0.0, 0.05, 0.2, 0.5, 0.75, 0.95, 1.0]:
            assert compute_risk_tier(prob) in valid_tiers, \
            f"Invalid tier for probability {prob}"

    def test_tier_boundary(self):
        # Lowest boundary = 'minimal', Highest boundary = 'critical'

        assert compute_risk_tier(0.0) == "minimal"
        assert compute_risk_tier(0.01) == "minimal"
        assert compute_risk_tier(0.96) == "critical"
        assert compute_risk_tier(1.0) == "critical"


    def test_tiers_increase_monotonically(self):
        # Higher probability should never have lower tier

        tier_order = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        probs = [0.0, 0.05, 0.2, 0.5, 0.94, 0.96, 1]
        tiers = [tier_order[compute_risk_tier(p)] for p in probs]
        assert tiers == sorted(tiers)


class TestComputeExpectedExposure:

    def test_zero_input_gives_zero_exposure(self):
        # probability = 0 or amount = 0 gives exposure = 0

        assert compute_expected_exposure(0.0, 143.8, "critical") == 0.0
        assert compute_expected_exposure(0.81, 0.0, "critical") == 0.0


    def test_higher_tier_gives_higher_exposure(self):
        # For same probability and amount, higher tier gives higher exposure

        prob = 0.97
        amount = 325.0
        assert (
            compute_expected_exposure(prob, amount, "minimal") <
            compute_expected_exposure(prob, amount, "low") <
            compute_expected_exposure(prob, amount, "medium") <
            compute_expected_exposure(prob, amount, "high") <
            compute_expected_exposure(prob, amount, "critical")
        )

    def test_exposure_formula(self):
        # critical (weight = 1.5) -> 0.9 * 100 * 1.5 = 135.00
        # minimal (weight = 0.1) -> 0.9 * 500 * 0.1 = 45.00

        assert compute_expected_exposure(0.9, 100.0, "critical") == 135.00
        assert compute_expected_exposure(0.9, 500, "minimal") == 45.00

    def test_unknown_tier_uses_default_weight(self):
        # when tier is not known, default weight should be used
        # (default weight = 1.0) -> 0.5 * 100 * 1.0 = 50.0

        exposure = compute_expected_exposure(0.5, 100, "unknown")
        assert exposure == 50.0
