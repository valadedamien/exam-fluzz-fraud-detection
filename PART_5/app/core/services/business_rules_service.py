from app.schemas import TransactionRequest, PredictionResponse
from config import settings

class BusinessRulesService:
    """Service pour appliquer les règles business"""
    
    def apply_rules(self, transaction: TransactionRequest, prediction: PredictionResponse) -> PredictionResponse:
        """Applique les règles business à une prédiction"""
        alerts = []
        requires_review = False
        
        # Montant élevé
        if transaction.Amount > settings.HIGH_AMOUNT_THRESHOLD:
            alerts.append(f"High amount: ${transaction.Amount:,.2f}")
            requires_review = True
        
        # Probabilité de fraude élevée mais pas détectée
        if prediction.fraud_probability > 0.3 and not prediction.is_fraud:
            alerts.append("Moderate fraud probability - consider review")
        
        # Très haute probabilité
        if prediction.fraud_probability > 0.8:
            alerts.append("Very high fraud probability")
            requires_review = True
        
        prediction.alerts = alerts
        prediction.requires_review = requires_review
        
        return prediction