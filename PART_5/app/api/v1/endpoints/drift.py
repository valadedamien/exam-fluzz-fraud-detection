from fastapi import APIRouter, HTTPException
from typing import Dict, List
import numpy as np
import logging

from app.core.services.drift_service import drift_detector

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/drift/reference")
async def set_reference_data(data: Dict[str, List[List[float]]]):
    """
    Configure les données de référence pour la détection de drift
    
    - **data**: {"reference_data": [[features], [features], ...]}
    """
    try:
        reference_data = np.array(data['reference_data'])
        drift_detector.set_reference_data(reference_data)
        
        return {
            "message": "Reference data configured successfully",
            "shape": reference_data.shape,
            "samples": len(reference_data)
        }
    except Exception as e:
        logger.error(f"Failed to set reference data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid reference data: {str(e)}")

@router.get("/drift/status")
async def get_drift_status():
    """
    Retourne l'état actuel de la détection de drift
    """
    try:
        summary = drift_detector.get_drift_summary()
        recent_alerts = drift_detector.get_recent_alerts(limit=5)
        
        return {
            "drift_status": summary,
            "recent_alerts": [
                {
                    "timestamp": alert.timestamp,
                    "drift_score": alert.drift_score,
                    "severity": alert.severity,
                    "message": alert.message
                } for alert in recent_alerts
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get drift status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get drift status: {str(e)}")

@router.get("/drift/alerts")
async def get_drift_alerts():
    """
    Retourne l'historique des alertes de drift
    """
    try:
        alerts = drift_detector.get_recent_alerts(limit=20)
        
        return {
            "total_alerts": len(alerts),
            "alerts": [
                {
                    "timestamp": alert.timestamp,
                    "drift_score": alert.drift_score, 
                    "severity": alert.severity,
                    "feature_index": alert.feature_index,
                    "message": alert.message
                } for alert in alerts
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.delete("/drift/reset")
async def reset_drift_detector():
    """
    Remet à zéro le détecteur de drift
    """
    try:
        drift_detector.current_window.clear()
        drift_detector.alerts_history.clear()
        drift_detector.last_drift_score = 0.0
        drift_detector.last_drift_check = 0
        
        return {"message": "Drift detector reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset drift detector: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset: {str(e)}")