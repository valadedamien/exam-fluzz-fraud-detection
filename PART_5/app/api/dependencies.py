from fastapi import Depends
from app.core import ModelService, PredictionService

# Instance globale du service de modèle
model_service = ModelService()

# Instance globale du service de prédiction
prediction_service = PredictionService(model_service)

def get_model_service() -> ModelService:
    """Dependency pour récupérer le service de modèle"""
    return model_service

def get_prediction_service() -> PredictionService:
    """Dependency pour récupérer le service de prédiction"""
    return prediction_service