from fastapi import APIRouter, HTTPException, Header, Depends
import logging

from app.schemas import ModelUpdateWebhook
from app.core import ModelService
from app.api.dependencies import get_model_service
from app.infrastructure.monitoring import model_reload_counter, current_model_version, model_age_seconds
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/model-updated")
async def model_updated_webhook(
    webhook_data: ModelUpdateWebhook,
    x_api_key: str = Header(...),
    model_service: ModelService = Depends(get_model_service)
):
    """
    Webhook appelé par Airflow lors de la mise à jour du modèle
    
    - **webhook_data**: Informations sur le nouveau modèle
    - **x_api_key**: Clé d'authentification
    """
    # Vérification de sécurité
    if x_api_key != settings.WEBHOOK_SECRET:
        logger.warning(f"Invalid webhook API key attempted")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    logger.info(f"🔔 Webhook received: Model v{webhook_data.model_version} in {webhook_data.stage}")
    logger.info(f"Metrics: Fraud Recall={webhook_data.metrics.get('test_fraud_recall', 'N/A'):.3f}")
    
    try:
        # Rechargement immédiat du modèle
        success = await model_service.load_specific_version(
            webhook_data.model_version, 
            webhook_data.stage
        )
        
        if success:
            # Métriques
            model_reload_counter.labels(trigger="webhook", success="true").inc()
            
            if model_service.current_version:
                try:
                    current_model_version.set(float(model_service.current_version))
                except ValueError:
                    current_model_version.set(1.0)
            
            model_age_seconds.set(0)  # Modèle tout frais
            
            logger.info(f"✅ Model v{webhook_data.model_version} loaded via webhook")
            
            return {
                "status": "success",
                "message": f"Model v{webhook_data.model_version} loaded successfully",
                "previous_version": model_service.previous_version,
                "current_version": model_service.current_version,
                "stage": model_service.current_stage,
                "metrics": webhook_data.metrics
            }
        else:
            raise Exception("Model loading returned false")
            
    except Exception as e:
        # En cas d'échec, la vérification périodique prendra le relais
        model_reload_counter.labels(trigger="webhook", success="false").inc()
        logger.error(f"❌ Webhook model update failed: {e}")
        
        return {
            "status": "error", 
            "message": str(e),
            "fallback": "Periodic check will attempt to load the model"
        }