import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

THRESHOLDS = config["risk"]["thresholds"]
SEVERITY_WEIGHTS = config["risk"]["severity_weights"]


def compute_risk_tier(fraud_probability: float) -> str:
    # A risk tier is assigned on the basis of fraud probability

    if fraud_probability < THRESHOLDS["low"]:
        return "minimal"
    elif fraud_probability < THRESHOLDS["medium"]:
        return "low"
    elif fraud_probability < THRESHOLDS["high"]:
        return "medium"
    elif fraud_probability < THRESHOLDS["critical"]:
        return "high"
    else:
        return "critical"


def compute_expected_exposure(
    fraud_probability: float, 
    amount: float, 
    risk_tier: str
) -> float:
    
    # Expected financial exposure for transaction
    severity_weight = SEVERITY_WEIGHTS.get(risk_tier, 1.0)
    exposure = fraud_probability * amount * severity_weight
    return round(exposure, 2)
    