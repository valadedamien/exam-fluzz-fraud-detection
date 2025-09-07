from pydantic import BaseModel, Field
from typing import Dict, Optional

class ModelUpdateWebhook(BaseModel):
    """Webhook de mise à jour du modèle depuis Airflow"""
    model_version: str
    stage: str = Field(..., pattern="^(Staging|Production)$")
    metrics: Dict[str, float]
    run_id: Optional[str] = None