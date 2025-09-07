import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import Optional, Dict, Any
import logging

from config import settings

logger = logging.getLogger(__name__)

class MLflowModelClient:
    """Client pour interagir avec MLflow Registry"""
    
    def __init__(self):
        self.client = MlflowClient(settings.MLFLOW_TRACKING_URI)
    
    def get_production_model(self):
        """Récupère le modèle en Production, sinon Staging"""
        try:
            # Essayer Production d'abord
            prod_models = self.client.get_latest_versions(
                name=settings.MODEL_NAME, 
                stages=["Production"]
            )
            if prod_models:
                return prod_models[0]
                
            # Fallback sur Staging
            staging_models = self.client.get_latest_versions(
                name=settings.MODEL_NAME, 
                stages=["Staging"]
            )
            if staging_models:
                return staging_models[0]
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting model from registry: {e}")
            return None
    
    def load_model(self, version: str):
        """Charge un modèle spécifique depuis MLflow"""
        model_uri = f"models:/{settings.MODEL_NAME}/{version}"
        return mlflow.sklearn.load_model(model_uri)
    
    def get_model_metrics(self, run_id: str) -> Dict[str, Any]:
        """Récupère les métriques d'un run"""
        run = self.client.get_run(run_id)
        return run.data.metrics
    
    def get_model_version_info(self, version: str):
        """Récupère les informations d'une version de modèle"""
        return self.client.get_model_version(settings.MODEL_NAME, version)
    
    def test_connection(self) -> bool:
        """Test la connexion à MLflow"""
        try:
            self.client.search_experiments()
            return True
        except Exception:
            return False