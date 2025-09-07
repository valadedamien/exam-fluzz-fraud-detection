import os
import joblib
import logging
from typing import Tuple, Optional

from config import settings

logger = logging.getLogger(__name__)

class FileModelLoader:
    """Loader pour charger les modèles depuis le volume partagé"""
    
    @staticmethod
    def load_model_and_scaler() -> Tuple[Optional[object], Optional[object]]:
        """Charge le modèle et scaler depuis les fichiers"""
        try:
            model_path = os.path.join(settings.SHARED_MODELS_PATH, "model.pkl")
            scaler_path = os.path.join(settings.SHARED_MODELS_PATH, "scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                logger.info("Loading model from shared volume...")
                
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                logger.info("✅ Model loaded from shared volume")
                return model, scaler
            else:
                raise FileNotFoundError("Model files not found in shared volume")
                
        except Exception as e:
            logger.error(f"Failed to load model from shared volume: {e}")
            raise
    
    @staticmethod 
    def model_files_exist() -> bool:
        """Vérifie si les fichiers de modèle existent"""
        model_path = os.path.join(settings.SHARED_MODELS_PATH, "model.pkl")
        scaler_path = os.path.join(settings.SHARED_MODELS_PATH, "scaler.pkl")
        return os.path.exists(model_path) and os.path.exists(scaler_path)