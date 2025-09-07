import time
import numpy as np
from typing import List
import logging

from app.schemas import TransactionRequest, PredictionResponse, BatchPredictionResponse
from app.core.services.model_service import ModelService
from app.core.services.business_rules_service import BusinessRulesService

logger = logging.getLogger(__name__)

class PredictionService:
    """Service de prédiction de fraude"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.business_rules = BusinessRulesService()
    
    def prepare_features(self, transaction: TransactionRequest) -> np.ndarray:
        """Prépare les features pour la prédiction"""
        # Convertir en format attendu par le modèle
        feature_names = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
        ]
        
        features = []
        for feature in feature_names:
            features.append(getattr(transaction, feature))
        
        return np.array(features).reshape(1, -1)
    
    async def predict_single(self, transaction: TransactionRequest) -> PredictionResponse:
        """Effectue une prédiction de fraude pour une transaction"""
        start_time = time.time()
        
        if not self.model_service.model:
            raise RuntimeError("No model loaded")
        
        try:
            # Préparer les features
            features = self.prepare_features(transaction)
            
            # Scaler si disponible
            if self.model_service.scaler:
                features = self.model_service.scaler.transform(features)
            
            # Prédiction
            prediction = self.model_service.model.predict(features)[0]
            probabilities = self.model_service.model.predict_proba(features)[0]
            
            # Construire la réponse
            processing_time = (time.time() - start_time) * 1000
            
            response = PredictionResponse(
                is_fraud=bool(prediction == 1),
                fraud_probability=float(probabilities[1] if len(probabilities) > 1 else prediction),
                confidence=float(max(probabilities) if len(probabilities) > 1 else abs(prediction - 0.5) + 0.5),
                processing_time_ms=round(processing_time, 2),
                model_version=self.model_service.current_version or "unknown",
                transaction_id=transaction.transaction_id
            )
            
            # Appliquer les règles business
            response = self.business_rules.apply_rules(transaction, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    async def predict_batch(self, transactions: List[TransactionRequest]) -> BatchPredictionResponse:
        """Effectue des prédictions pour un lot de transactions"""
        start_time = time.time()
        
        predictions = []
        fraud_count = 0
        
        for transaction in transactions:
            prediction = await self.predict_single(transaction)
            predictions.append(prediction)
            
            if prediction.is_fraud:
                fraud_count += 1
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(transactions),
            total_fraud_detected=fraud_count,
            batch_processing_time_ms=round(processing_time, 2)
        )