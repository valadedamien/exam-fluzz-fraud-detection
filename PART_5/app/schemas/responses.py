from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    is_fraud: bool = Field(..., description="True si fraude détectée")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probabilité de fraude (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Confiance du modèle")
    processing_time_ms: float = Field(..., description="Temps de traitement en ms")
    model_version: str = Field(..., description="Version du modèle utilisé")
    transaction_id: Optional[str] = Field(None, description="ID de la transaction")
    
    # Alertes business
    alerts: List[str] = Field(default_factory=list, description="Alertes business (montant élevé, etc.)")
    requires_review: bool = Field(False, description="Nécessite une revue manuelle")

class BatchPredictionResponse(BaseModel):
    """Réponse de prédiction par lot"""
    predictions: List[PredictionResponse]
    total_processed: int
    total_fraud_detected: int
    batch_processing_time_ms: float

class HealthResponse(BaseModel):
    """Réponse du health check"""
    model_config = {"protected_namespaces": ()}
    
    status: str
    model_loaded: bool
    model_version: Optional[str]
    model_stage: Optional[str]
    last_update: Optional[datetime]
    mlflow_connection: bool
    uptime_seconds: float