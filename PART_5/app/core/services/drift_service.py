import time
import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import deque
import logging
from dataclasses import dataclass

from app.schemas import TransactionRequest

logger = logging.getLogger(__name__)

@dataclass
class DriftAlert:
    """Alerte de drift détectée"""
    timestamp: float
    drift_score: float
    severity: str
    feature_index: Optional[int] = None
    message: str = ""

class RealTimeDriftDetector:
    """
    Détecteur de drift temps réel optimisé pour l'API
    Utilise des fenêtres glissantes pour détecter le drift rapidement
    """
    
    def __init__(self, 
                 window_size: int = 1000,
                 reference_data: Optional[np.ndarray] = None,
                 drift_threshold: float = 0.15,
                 alert_threshold: float = 0.25):
        
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.alert_threshold = alert_threshold
        
        # Fenêtre glissante des dernières transactions
        self.current_window = deque(maxlen=window_size)
        
        # Données de référence (échantillon du dataset d'entraînement)
        self.reference_data = reference_data
        self.reference_stats = None
        
        # Cache des calculs de drift
        self.last_drift_check = 0
        self.last_drift_score = 0.0
        self.drift_check_interval = 100  # Vérifier le drift tous les N échantillons
        
        # Historique des alertes
        self.alerts_history = deque(maxlen=50)
        
        if reference_data is not None:
            self._compute_reference_stats()
    
    def set_reference_data(self, reference_data: np.ndarray):
        """Définit les données de référence"""
        self.reference_data = reference_data.copy()
        self._compute_reference_stats()
        logger.info(f"Reference data set: {reference_data.shape}")
    
    def _compute_reference_stats(self):
        """Précalcule les statistiques des données de référence"""
        if self.reference_data is None:
            return
            
        self.reference_stats = {
            'mean': np.mean(self.reference_data, axis=0),
            'std': np.std(self.reference_data, axis=0),
            'quantiles': {
                'q25': np.percentile(self.reference_data, 25, axis=0),
                'q50': np.percentile(self.reference_data, 50, axis=0), 
                'q75': np.percentile(self.reference_data, 75, axis=0)
            }
        }
    
    def add_transaction(self, transaction: TransactionRequest) -> Dict:
        """
        Ajoute une transaction à la fenêtre et vérifie le drift si nécessaire
        """
        # Convertir la transaction en features
        features = self._transaction_to_features(transaction)
        self.current_window.append(features)
        
        drift_result = {
            'drift_score': self.last_drift_score,
            'is_drift_detected': self.last_drift_score > self.drift_threshold,
            'alert_triggered': False,
            'window_size': len(self.current_window),
            'time_since_last_check': len(self.current_window) - self.last_drift_check
        }
        
        # Vérifier le drift périodiquement ou si fenêtre pleine
        should_check = (
            len(self.current_window) >= self.window_size or
            (len(self.current_window) - self.last_drift_check) >= self.drift_check_interval
        )
        
        if should_check and self.reference_data is not None:
            drift_result = self._check_drift()
            self.last_drift_check = len(self.current_window)
        
        return drift_result
    
    def _transaction_to_features(self, transaction: TransactionRequest) -> np.ndarray:
        """Convertit une transaction en vecteur de features"""
        feature_names = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
        ]
        
        features = []
        for feature in feature_names:
            features.append(getattr(transaction, feature))
        
        return np.array(features)
    
    def _check_drift(self) -> Dict:
        """Vérifie le drift entre la fenêtre courante et les données de référence"""
        if len(self.current_window) < 50:  # Minimum d'échantillons
            return {
                'drift_score': 0.0,
                'is_drift_detected': False,
                'alert_triggered': False,
                'message': 'Insufficient samples for drift detection',
                'window_size': len(self.current_window)
            }
        
        current_data = np.array(list(self.current_window))
        drift_score = self._calculate_drift_score(current_data)
        
        self.last_drift_score = drift_score
        is_drift = drift_score > self.drift_threshold
        alert_triggered = drift_score > self.alert_threshold
        
        # Créer une alerte si nécessaire
        if alert_triggered:
            alert = DriftAlert(
                timestamp=time.time(),
                drift_score=drift_score,
                severity='HIGH' if drift_score > 0.4 else 'MEDIUM',
                message=f"Drift critique détecté: score {drift_score:.3f}"
            )
            self.alerts_history.append(alert)
            logger.warning(f"🚨 {alert.message}")
        
        return {
            'drift_score': drift_score,
            'is_drift_detected': is_drift,
            'alert_triggered': alert_triggered,
            'window_size': len(self.current_window),
            'severity': 'HIGH' if drift_score > 0.4 else ('MEDIUM' if drift_score > 0.25 else 'LOW')
        }
    
    def _calculate_drift_score(self, current_data: np.ndarray) -> float:
        """
        Calcul rapide du drift score en utilisant des statistiques précalculées
        """
        try:
            if self.reference_stats is None:
                return 0.0
            
            # Calcul simplifié basé sur les moyennes et écarts-types
            current_mean = np.mean(current_data, axis=0)
            current_std = np.std(current_data, axis=0)
            
            ref_mean = self.reference_stats['mean']
            ref_std = self.reference_stats['std']
            
            # Drift basé sur la différence normalisée des moyennes
            mean_diff = np.abs(current_mean - ref_mean)
            normalized_diff = mean_diff / (ref_std + 1e-6)  # Éviter division par 0
            
            # Score de drift global (moyenne des drifts par feature)
            drift_score = np.mean(normalized_diff)
            
            # Normaliser entre 0 et 1
            drift_score = min(1.0, drift_score / 3.0)  # Diviseur empirique
            
            return float(drift_score)
            
        except Exception as e:
            logger.error(f"Error calculating drift score: {e}")
            return 0.0
    
    def get_recent_alerts(self, limit: int = 10) -> List[DriftAlert]:
        """Retourne les alertes récentes"""
        return list(self.alerts_history)[-limit:]
    
    def get_drift_summary(self) -> Dict:
        """Retourne un résumé de l'état du drift"""
        recent_alerts = len([a for a in self.alerts_history if time.time() - a.timestamp < 3600])  # 1h
        
        return {
            'current_drift_score': self.last_drift_score,
            'is_drift_detected': self.last_drift_score > self.drift_threshold,
            'window_size': len(self.current_window),
            'max_window_size': self.window_size,
            'recent_alerts_1h': recent_alerts,
            'total_alerts': len(self.alerts_history),
            'reference_data_available': self.reference_data is not None,
            'last_check_samples_ago': len(self.current_window) - self.last_drift_check
        }

# Instance globale pour l'API
drift_detector = RealTimeDriftDetector()