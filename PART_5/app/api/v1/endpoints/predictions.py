import time
from fastapi import APIRouter, HTTPException, Depends
import logging

from app.schemas import TransactionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from app.core import PredictionService
from app.api.dependencies import get_prediction_service
from app.infrastructure.monitoring import (
    prediction_counter, prediction_latency, batch_size_histogram,
    fraud_detection_rate, high_amount_transactions, drift_score_gauge,
    drift_window_size, drift_alerts_counter, drift_checks_counter
)
from config import settings
from app.core.services.drift_service import drift_detector

logger = logging.getLogger(__name__)
router = APIRouter()

# Variables globales pour tracking
total_predictions = 0
total_fraud_detected = 0

@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    transaction: TransactionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Prédiction de fraude pour une transaction unique
    
    - **transaction**: Données de la transaction avec toutes les features V1-V28, Amount, Time
    - **returns**: Prédiction avec probabilité de fraude et métadonnées
    """
    global total_predictions, total_fraud_detected
    
    try:
        # Vérifier que le modèle est chargé
        if not prediction_service.model_service.model:
            raise HTTPException(
                status_code=503, 
                detail="No model loaded - service unavailable"
            )
        
        # Effectuer la prédiction
        with prediction_latency.time():
            prediction = await prediction_service.predict_single(transaction)
        
        # Mise à jour des métriques
        result = "fraud" if prediction.is_fraud else "normal"
        prediction_counter.labels(
            result=result, 
            model_version=prediction_service.model_service.current_version or "unknown"
        ).inc()
        
        # Métriques business
        total_predictions += 1
        if prediction.is_fraud:
            total_fraud_detected += 1
        
        if total_predictions > 0:
            fraud_detection_rate.set(total_fraud_detected / total_predictions)
        
        # Alertes montant élevé
        if transaction.Amount > settings.HIGH_AMOUNT_THRESHOLD:
            high_amount_transactions.inc()
        
        # Détection de drift temps réel
        drift_result = drift_detector.add_transaction(transaction)
        
        # Mise à jour des métriques de drift
        drift_score_gauge.set(drift_result['drift_score'])
        drift_window_size.set(drift_result['window_size'])
        
        # Compter les vérifications de drift
        if 'time_since_last_check' in drift_result and drift_result['time_since_last_check'] == 0:
            drift_result_label = 'drift' if drift_result['is_drift_detected'] else 'normal'
            drift_checks_counter.labels(result=drift_result_label).inc()
            
            # Alertes de drift
            if drift_result['alert_triggered']:
                severity = drift_result.get('severity', 'MEDIUM').lower()
                drift_alerts_counter.labels(severity=severity).inc()
                logger.warning(f"🚨 DRIFT ALERT: {drift_result['severity']} - Score: {drift_result['drift_score']:.3f}")
        
        logger.info(f"Prediction: {result} (prob: {prediction.fraud_probability:.3f}, "
                   f"amount: ${transaction.Amount:,.2f}, time: {prediction.processing_time_ms:.1f}ms, "
                   f"drift: {drift_result['drift_score']:.3f})")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict_fraud(
    batch_request: BatchPredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Prédiction de fraude pour un lot de transactions
    
    - **batch_request**: Liste de transactions (max 100)
    - **returns**: Prédictions pour toutes les transactions
    """
    global total_predictions, total_fraud_detected
    
    try:
        if not prediction_service.model_service.model:
            raise HTTPException(
                status_code=503, 
                detail="No model loaded - service unavailable"
            )
        
        transactions = batch_request.transactions
        batch_size = len(transactions)
        
        # Métrique de taille de batch
        batch_size_histogram.observe(batch_size)
        
        logger.info(f"Processing batch of {batch_size} transactions")
        
        # Effectuer les prédictions
        batch_response = await prediction_service.predict_batch(transactions)
        
        # Mise à jour des métriques globales
        total_predictions += batch_response.total_processed
        total_fraud_detected += batch_response.total_fraud_detected
        
        logger.info(f"Batch processed: {batch_response.total_fraud_detected}/{batch_size} fraud detected "
                   f"in {batch_response.batch_processing_time_ms:.1f}ms")
        
        return batch_response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")