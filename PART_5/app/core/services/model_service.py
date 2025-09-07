import asyncio
import time
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from app.infrastructure.mlflow import MLflowModelClient
from app.infrastructure.storage import FileModelLoader
from config import settings

logger = logging.getLogger(__name__)

class ModelService:
    """Service de gestion des modèles avec stratégie hybride de mise à jour"""
    
    def __init__(self):
        self.mlflow_client = MLflowModelClient()
        self.current_version: Optional[str] = None
        self.previous_version: Optional[str] = None
        self.current_stage: Optional[str] = None
        self.model = None
        self.scaler = None
        self.last_update: Optional[datetime] = None
        self.loaded_at: float = 0
        self.last_check: float = 0
        self.model_metrics: Dict[str, float] = {}
        
        # Lock pour éviter les chargements simultanés
        self._loading_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialisation du service au démarrage"""
        logger.info("Initializing ModelService...")
        await self.load_initial_model()
        
    async def load_initial_model(self):
        """Charge le modèle initial au démarrage"""
        try:
            # 1. Essayer de charger depuis MLflow
            await self.load_latest_from_mlflow()
            
        except Exception as e:
            logger.warning(f"Failed to load from MLflow: {e}")
            try:
                # 2. Fallback: charger depuis le volume partagé
                await self.load_from_shared_volume()
            except Exception as e2:
                logger.error(f"Failed to load from shared volume: {e2}")
                raise RuntimeError("No model could be loaded at startup")
    
    async def load_latest_from_mlflow(self):
        """Charge le dernier modèle depuis MLflow Registry"""
        async with self._loading_lock:
            try:
                # Récupère la dernière version en Production, sinon Staging
                latest_model = self.mlflow_client.get_production_model()
                
                if latest_model and latest_model.version != self.current_version:
                    logger.info(f"Loading model v{latest_model.version} from MLflow...")
                    
                    # Charger le modèle
                    new_model = self.mlflow_client.load_model(latest_model.version)
                    
                    # Charger les métriques associées
                    metrics = self.mlflow_client.get_model_metrics(latest_model.run_id)
                    
                    # Mise à jour atomique
                    self.previous_version = self.current_version
                    self.model = new_model
                    self.current_version = latest_model.version
                    self.current_stage = latest_model.current_stage
                    self.model_metrics = metrics
                    self.last_update = datetime.now()
                    self.loaded_at = time.time()
                    
                    logger.info(f"✅ Model v{latest_model.version} loaded successfully")
                    logger.info(f"Fraud Recall: {metrics.get('test_fraud_recall', 'N/A'):.3f}")
                    logger.info(f"Frauds Missed: {metrics.get('frauds_missed_count', 'N/A')}")
                    
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to load model from MLflow: {e}")
                raise
                
    async def load_from_shared_volume(self):
        """Fallback: charge le modèle depuis le volume partagé"""
        async with self._loading_lock:
            try:
                model, scaler = FileModelLoader.load_model_and_scaler()
                
                # Mise à jour
                self.model = model
                self.scaler = scaler
                self.current_version = f"shared_volume_{int(time.time())}"
                self.current_stage = "Unknown"
                self.last_update = datetime.now()
                self.loaded_at = time.time()
                
                logger.info("✅ Model loaded from shared volume")
                return True
                    
            except Exception as e:
                logger.error(f"Failed to load model from shared volume: {e}")
                raise
    
    async def periodic_model_check(self):
        """Tâche background de vérification périodique"""
        logger.info(f"Starting periodic model check (every {settings.MODEL_CHECK_INTERVAL}s)")
        
        while True:
            try:
                await asyncio.sleep(settings.MODEL_CHECK_INTERVAL)
                
                logger.debug("Performing periodic model check...")
                if await self.check_for_updates():
                    logger.info("Model updated via periodic check")
                    
            except Exception as e:
                logger.error(f"Periodic check failed: {e}")
                # Attendre moins longtemps en cas d'erreur
                await asyncio.sleep(60)
    
    async def check_for_updates(self) -> bool:
        """Vérifie si une mise à jour est disponible"""
        try:
            latest_model = self.mlflow_client.get_production_model()
            
            if latest_model and latest_model.version != self.current_version:
                logger.info(f"New model version detected: v{latest_model.version}")
                await self.load_latest_from_mlflow()
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
            return False
    
    async def load_specific_version(self, version: str, stage: str = None) -> bool:
        """Charge une version spécifique du modèle"""
        async with self._loading_lock:
            try:
                logger.info(f"Loading specific model version: v{version}")
                
                new_model = self.mlflow_client.load_model(version)
                
                # Récupérer les métadonnées
                model_version = self.mlflow_client.get_model_version_info(version)
                metrics = self.mlflow_client.get_model_metrics(model_version.run_id)
                
                # Mise à jour
                self.previous_version = self.current_version
                self.model = new_model
                self.current_version = version
                self.current_stage = stage or model_version.current_stage
                self.model_metrics = metrics
                self.last_update = datetime.now()
                self.loaded_at = time.time()
                
                logger.info(f"✅ Model v{version} loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model v{version}: {e}")
                raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle actuel"""
        return {
            "version": self.current_version,
            "stage": self.current_stage,
            "loaded_at": self.last_update.isoformat() if self.last_update else None,
            "model_age_seconds": time.time() - self.loaded_at if self.loaded_at else None,
            "metrics": self.model_metrics,
            "mlflow_uri": settings.MLFLOW_TRACKING_URI
        }