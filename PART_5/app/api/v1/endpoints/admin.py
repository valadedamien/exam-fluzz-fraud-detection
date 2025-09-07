from fastapi import APIRouter, HTTPException, Header, Depends
import logging

from app.core import ModelService
from app.api.dependencies import get_model_service
from app.infrastructure.monitoring import model_reload_counter, current_model_version, model_age_seconds
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/reload-model")
async def manual_reload_model(
    target_version: str = None,
    stage: str = None,
    x_admin_key: str = Header(...),
    model_service: ModelService = Depends(get_model_service)
):
    """
    Rechargement manuel du modèle (administration)
    
    - **target_version**: Version spécifique à charger (optionnel)
    - **stage**: Stage du modèle (optionnel)
    - **x_admin_key**: Clé d'administration
    """
    if x_admin_key != settings.ADMIN_SECRET:
        logger.warning(f"Invalid admin API key attempted")
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        logger.info(f"🔧 Manual reload requested - Version: {target_version or 'latest'}")
        
        if target_version:
            # Charger version spécifique
            success = await model_service.load_specific_version(target_version, stage)
        else:
            # Charger la dernière version disponible
            success = await model_service.check_for_updates()
            if not success:
                # Force reload from MLflow
                await model_service.load_latest_from_mlflow()
                success = True
        
        if success:
            model_reload_counter.labels(trigger="manual", success="true").inc()
            
            # Mise à jour des métriques
            if model_service.current_version:
                try:
                    current_model_version.set(float(model_service.current_version))
                except ValueError:
                    current_model_version.set(1.0)
            
            model_age_seconds.set(0)
            
            logger.info(f"✅ Manual reload successful: v{model_service.current_version}")
            
            return {
                "status": "success",
                "current_version": model_service.current_version,
                "stage": model_service.current_stage,
                "loaded_at": model_service.last_update.isoformat(),
                "model_metrics": model_service.model_metrics
            }
        else:
            raise Exception("No updates available or loading failed")
            
    except Exception as e:
        model_reload_counter.labels(trigger="manual", success="false").inc()
        logger.error(f"❌ Manual reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")

@router.post("/rollback")
async def rollback_model(
    target_version: str,
    x_admin_key: str = Header(...),
    model_service: ModelService = Depends(get_model_service)
):
    """
    Rollback vers une version précédente du modèle
    
    - **target_version**: Version vers laquelle faire le rollback
    - **x_admin_key**: Clé d'administration
    """
    if x_admin_key != settings.ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        logger.warning(f"🔄 ROLLBACK requested to version {target_version}")
        
        await model_service.load_specific_version(target_version)
        
        model_reload_counter.labels(trigger="rollback", success="true").inc()
        
        logger.warning(f"⚠️ ROLLBACK completed: Now using v{model_service.current_version}")
        
        return {
            "status": "rollback_success",
            "version": model_service.current_version,
            "stage": model_service.current_stage,
            "rollback_time": model_service.last_update.isoformat(),
            "warning": "System rolled back to previous version"
        }
        
    except Exception as e:
        model_reload_counter.labels(trigger="rollback", success="false").inc()
        logger.error(f"❌ Rollback failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rollback failed: {str(e)}")

@router.get("/config")
async def debug_config(
    x_admin_key: str = Header(...),
):
    """Configuration debug (admin seulement)"""
    if x_admin_key != settings.ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    return {
        "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
        "model_name": settings.MODEL_NAME,
        "check_interval": settings.MODEL_CHECK_INTERVAL,
        "shared_models_path": settings.SHARED_MODELS_PATH,
        "high_amount_threshold": settings.HIGH_AMOUNT_THRESHOLD,
        "fraud_probability_threshold": settings.FRAUD_PROBABILITY_THRESHOLD
    }