from fastapi import APIRouter

from app.api.v1.endpoints import predictions_router, health_router, admin_router, webhooks_router
from app.api.v1.endpoints.drift import router as drift_router

api_router = APIRouter()

# Endpoints de prédiction
api_router.include_router(predictions_router, tags=["predictions"])

# Endpoints de santé et info
api_router.include_router(health_router, tags=["health"])

# Endpoints admin
api_router.include_router(admin_router, prefix="/admin", tags=["admin"])

# Endpoints webhooks  
api_router.include_router(webhooks_router, prefix="/webhook", tags=["webhooks"])

# Endpoints drift monitoring
api_router.include_router(drift_router, prefix="/monitoring", tags=["drift"])