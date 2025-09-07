import time
from fastapi import APIRouter, HTTPException, Depends

from app.schemas import HealthResponse
from app.core import ModelService
from app.api.dependencies import get_model_service

router = APIRouter()

# App start time pour calcul uptime
app_start_time = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check(
    model_service: ModelService = Depends(get_model_service)
):
    """
    Health check de l'API
    
    Vérifie l'état du service, la connexion MLflow et le modèle chargé
    """
    try:
        # Test de connexion MLflow
        mlflow_connection = model_service.mlflow_client.test_connection()
        
        # Calcul de l'uptime
        uptime = time.time() - app_start_time
        
        return HealthResponse(
            status="healthy" if model_service.model else "degraded",
            model_loaded=model_service.model is not None,
            model_version=model_service.current_version,
            model_stage=model_service.current_stage,
            last_update=model_service.last_update,
            mlflow_connection=mlflow_connection,
            uptime_seconds=round(uptime, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/model/info")
async def get_model_info(
    model_service: ModelService = Depends(get_model_service)
):
    """
    Informations détaillées sur le modèle actuel
    
    Inclut les métriques de performance, version, et métadonnées
    """
    if not model_service.current_version:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    model_info = model_service.get_model_info()
    
    # Ajouter des statistiques en temps réel
    from app.api.v1.endpoints.predictions import total_predictions, total_fraud_detected
    
    model_info.update({
        "runtime_stats": {
            "total_predictions": total_predictions,
            "total_fraud_detected": total_fraud_detected,
            "fraud_detection_rate": total_fraud_detected / max(total_predictions, 1),
            "api_uptime_seconds": time.time() - app_start_time
        }
    })
    
    return model_info