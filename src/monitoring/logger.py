# Logging monitoring events into audit_logs table

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def _write(event_type: str, details: dict, transaction_id: str = None) -> None:
    # Structuring dict into JSON and writing in one row

    try:
        from src.database.db import log_audit_event
        details_str = json.dumps(details, default=str)

        log_audit_event(
            event_type=event_type,
            details=details_str,
            transaction_id=transaction_id
        )
        logger.info(f"[audit] {event_type} | {details_str}")

    except Exception as e:
        logger.warning(f"[audit] Failed to write event '{event_type}' to audit_logs : {e}")


def log_drift_detected(
    drift_share: float,
    n_drifted: int,
    n_features: int,
    drifted_features: list[str],
    reference_period: str,
    current_period: str
) -> None:
    """
    For logging drift if it is detected by Evidently AI 

    drift_share : fraction of drifted features (0.0 to 1.0)
    n_drifted : number of drifted features
    n_features : total number of features
    drifted_features : list of drifted feature names 
    reference_period : previous version training period, eg- "1991-2010"
    current_period : new version data period, eg- "2015-2020"
    """

    _write(
        event_type="drift_detected",
        details={
            "drift_share": round(drift_share, 4),
            "n_drifted": n_drifted,
            "n_features": n_features,
            "drifted_features": drifted_features,
            "reference_period": reference_period,
            "current_period": current_period,
            "detected_at": datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M:%S UTC")
        }
    )


def log_drift_report_generated(
    output_path: str,
    reference_period: str,
    current_period: str,
    n_reference_rows: int,
    n_current_rows: int
) -> None:
    # For logging HTML drift report written to disc
    # Drift report detects or does not detect drift

    _write(
        event_type="drift_report_generated",
        details={
            "output_path" : output_path,
            "reference_period" : reference_period,
            "current_period" : current_period,
            "n_reference_rows" : n_reference_rows,
            "n_current_rows": n_current_rows,
            "generated_at": datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M:%S UTC")
        }
    )


def log_model_promoted(
    from_version: str,
    to_version: str,
    artifact_path: str,
    triggered_by: str = "api"
) -> None:
    # Called after app.state.model is changed to new version 
    # triggered by can be - "api", "dashboard", "script"

    _write(
        event_type="model_promoted",
        details={
            "from_version": from_version,
            "to_version": to_version,
            "artifact_path": artifact_path,
            "triggered_by": triggered_by,
            "promoted_at": datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M:%S UTC")
        }
    )


def log_model_rolled_back(
    from_version: str,
    to_version: str,
    artifact_path: str,
    triggered_by: str = "api"
) -> None:
    # Called after app.state.model is changed to older version

    _write(
        event_type="model_rolled_back",
        details={
            "from_version": from_version,
            "to_version": to_version,
            "artifact_path": artifact_path,
            "triggered_by": triggered_by,
            "rolled_back_at": datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M:%S UTC")
        }
    )


def log_model_switch_failed(
    requested_version: str,
    current_version: str,
    reason: str
) -> None:
    # Called when promotion or rollback fails
    # The active model version stays same as before

    _write(
        event_type="model_switch_failed",
        details={
            "requested_version": requested_version,
            "current_version": current_version,
            "reason": reason,
            "failed_at": datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M:%S UTC")
        }
    )


def log_performance_summary(
    model_version: str,
    period: str,
    total_predictions: int,
    total_alerts: int,
    avg_latency_ms: float
) -> None:
    
    alert_rate = (
        round(total_alerts / total_predictions * 100, 4)
        if total_predictions > 0
        else 0.00
    )

    _write(
        event_type="performance_summary",
        details={
            "model_version": model_version,
            "period": period,
            "total_predictions": total_predictions,
            "total_alerts": total_alerts,
            "alert_rate": alert_rate,
            "avg_latency_ms": avg_latency_ms,
            "logged_at": datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M:%S UTC")
        }
    )