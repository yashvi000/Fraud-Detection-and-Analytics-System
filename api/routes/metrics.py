import logging
from fastapi import APIRouter
from api.schemas import MetricsOutput
from src.database.db import get_metrics

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/metrics", response_model=MetricsOutput)
def get_dashboard_metrics():
    # Fetching metrics from predictions table for dashboard KPI

    metrics = get_metrics()

    logger.info(
        f"GET /metrics | "
        f"total : {metrics.get('total_predictions', 0):,} | "
        f"alerts : {metrics.get('total_alerts'):,} | "
        f"rate : {metrics.get('alert_rate', 0)}%"
    )

    return MetricsOutput(**metrics)