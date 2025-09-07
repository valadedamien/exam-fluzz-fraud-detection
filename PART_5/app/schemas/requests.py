from pydantic import BaseModel, Field
from typing import List, Optional

class TransactionRequest(BaseModel):
    """Modèle de validation pour une transaction à prédire"""
    # Features PCA du dataset creditcard.csv
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    
    # Features originales
    Amount: float = Field(..., ge=0, description="Transaction amount")
    Time: float = Field(..., ge=0, description="Seconds elapsed since first transaction")
    
    # Métadonnées optionnelles
    transaction_id: Optional[str] = Field(None, description="Transaction ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "V1": -1.3598071336738,
                "V2": -0.0727811733098497,
                "V3": 2.5363467416473,
                "V4": 1.3781155464166,
                "V5": -0.3383207699688,
                "V6": 0.4623878932669,
                "V7": 0.2395515152269,
                "V8": 0.0986979012610,
                "V9": 0.3637870005694,
                "V10": 0.0907941719789,
                "V11": -0.551599533260,
                "V12": -0.617801717057,
                "V13": -0.991389847235,
                "V14": -0.311169353699,
                "V15": 1.4681770385523,
                "V16": -0.470400525259,
                "V17": 0.2077821225275,
                "V18": 0.0257905801960,
                "V19": 0.4039934058309,
                "V20": 0.2514148072509,
                "V21": -0.018306777944,
                "V22": 0.2773500011259,
                "V23": -0.110473910188,
                "V24": 0.0669280749146,
                "V25": 0.1285394016206,
                "V26": -0.189114843888,
                "V27": 0.1335558617769,
                "V28": -0.021053053102,
                "Amount": 149.62,
                "Time": 0,
                "transaction_id": "txn_123456789"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Requête de prédiction par lot"""
    transactions: List[TransactionRequest] = Field(..., min_items=1, max_items=100)