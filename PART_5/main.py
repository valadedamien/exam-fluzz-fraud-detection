import asyncio
import time
from typing import Dict, Any
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import generate_latest

from app.api import api_router, get_model_service
from app.infrastructure.monitoring import registry, api_requests_total, current_model_version
from config import settings

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialisation FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="API de détection de fraude temps réel avec mise à jour automatique des modèles",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
app_start_time = time.time()

# === MIDDLEWARE DE MONITORING ===

@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Vérifier les mises à jour de modèle périodiquement
    model_service = get_model_service()
    if hasattr(model_service, 'last_check'):
        if time.time() - model_service.last_check > settings.MODEL_CHECK_INTERVAL:
            try:
                await model_service.check_for_updates()
                model_service.last_check = time.time()
            except Exception as e:
                logger.warning(f"Background model check failed: {e}")
    
    # Traitement de la requête
    response = await call_next(request)
    
    # Métriques
    api_requests_total.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()
    
    return response

# === ÉVÉNEMENTS DE DÉMARRAGE/ARRÊT ===

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    logger.info("🚀 Starting Fraud Detection API...")
    
    try:
        # Initialiser le service de modèles
        model_service = get_model_service()
        await model_service.initialize()
        
        # Démarrer la vérification périodique en arrière-plan
        asyncio.create_task(model_service.periodic_model_check())
        
        # Initialiser les métriques
        if model_service.current_version:
            try:
                current_model_version.set(float(model_service.current_version))
            except ValueError:
                current_model_version.set(1.0)  # Fallback pour versions non-numériques
        
        logger.info("✅ Fraud Detection API started successfully")
        logger.info(f"📊 Model loaded: v{model_service.current_version} ({model_service.current_stage})")
        
    except Exception as e:
        logger.error(f"❌ Failed to start API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt"""
    logger.info("🛑 Shutting down Fraud Detection API...")

# === ENDPOINTS ===

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Endpoint racine avec informations de l'API"""
    model_service = get_model_service()
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "status": "operational",
        "model_version": model_service.current_version,
        "model_stage": model_service.current_stage,
        "uptime_seconds": round(time.time() - app_start_time, 2),
        "endpoints": {
            "predict": "POST /predict - Single transaction prediction",
            "batch_predict": "POST /predict/batch - Batch predictions",
            "health": "GET /health - Health check",
            "model_info": "GET /model/info - Current model information",
            "metrics": "GET /metrics - Prometheus metrics"
        }
    }

@app.get("/metrics")
async def get_prometheus_metrics():
    """
    Métriques Prometheus pour monitoring
    
    Format texte compatible avec Prometheus/Grafana
    """
    return Response(
        generate_latest(registry),
        media_type="text/plain"
    )

# Inclusion du router API
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)