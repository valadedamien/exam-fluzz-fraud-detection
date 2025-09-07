import os
from typing import Optional

class Settings:
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    MODEL_NAME: str = "MLPClassifier"
    
    # API Configuration  
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "fraud-detection-webhook-secret")
    ADMIN_SECRET: str = os.getenv("ADMIN_SECRET", "fraud-detection-admin-secret")
    
    # Model Update Configuration
    MODEL_CHECK_INTERVAL: int = int(os.getenv("MODEL_CHECK_INTERVAL", "300"))  # 5 minutes
    SHARED_MODELS_PATH: str = os.getenv("SHARED_MODELS_PATH", "/shared/models")
    
    # Prometheus Configuration
    PROMETHEUS_PUSHGATEWAY_URL: str = os.getenv("PROMETHEUS_PUSHGATEWAY_URL", "http://localhost:9091")
    
    # Business Rules
    HIGH_AMOUNT_THRESHOLD: float = 10000.0
    FRAUD_PROBABILITY_THRESHOLD: float = 0.5

settings = Settings()