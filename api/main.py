import os
import yaml
import logging
import joblib
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from api.routes import predict, alerts, metrics

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Setting up logger
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Loading config and paths
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

ARTIFACTS_PATH = PROJECT_ROOT / config["artifacts"]["model_dir"]

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading all artifacts once (at start), storing in app.state

    logger.info("Loading model artifacts ...")

    app.state.model = joblib.load(ARTIFACTS_PATH / "lgbm_v1.pkl")
    app.state.mcc_freq = joblib.load(ARTIFACTS_PATH / "mcc_encoding.pkl")
    app.state.cold_start = joblib.load(ARTIFACTS_PATH / "cold_start_fill_values.pkl")

    app.state.config = config
    app.state.threshold = config["model"]["threshold"]
    app.state.feature_cols = config["feature_cols"]

    logger.info("Model Artifacts loaded")
    logger.info(f"Threshold : {app.state.threshold}")
    logger.info(f"Feature Columns : {len(app.state.feature_cols)}")

    # Shutdown
    yield
    logger.info("Shutting down FastAPI")


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection and risk scoring",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(predict.router)
app.include_router(alerts.router)
app.include_router(metrics.router)


@app.get("/health")
def health():
    # Checking endpoint health
    return {"status" : "ok"}