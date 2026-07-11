import logging
from fastapi import APIRouter, Query
from api.schemas import AlertOutput, AlertRecord
from src.database.db import get_recent_alerts

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/alerts", response_model=AlertOutput)
def get_alerts(limit: int = Query(default=50, ge=1, le=500)):
    # Fetching recent fraud alert records

    rows = get_recent_alerts(limit=limit)
    alerts = [AlertRecord(**row) for row in rows]

    logger.info(f"GET /alerts | limit : {limit} | returned : {len(alerts)}")
    
    return AlertOutput(alerts=alerts, total=len(alerts))