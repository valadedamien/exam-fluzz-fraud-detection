from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# Registry séparée pour éviter les conflits
registry = CollectorRegistry()

# Métriques de l'API
api_requests_total = Counter(
    'fraud_api_requests_total', 
    'Total API requests', 
    ['endpoint', 'method', 'status'], 
    registry=registry
)

prediction_counter = Counter(
    'fraud_predictions_total', 
    'Total predictions made', 
    ['result', 'model_version'], 
    registry=registry
)

prediction_latency = Histogram(
    'fraud_prediction_duration_seconds', 
    'Prediction latency in seconds',
    registry=registry
)

batch_size_histogram = Histogram(
    'fraud_batch_size', 
    'Batch prediction sizes',
    registry=registry
)

# Métriques du modèle
current_model_version = Gauge(
    'fraud_model_version', 
    'Current model version', 
    registry=registry
)

model_age_seconds = Gauge(
    'fraud_model_age_seconds', 
    'Age of current model in seconds',
    registry=registry
)

model_reload_counter = Counter(
    'fraud_model_reloads_total', 
    'Model reload events', 
    ['trigger', 'success'], 
    registry=registry
)

# Métriques business
fraud_detection_rate = Gauge(
    'fraud_api_detection_rate', 
    'Real-time fraud detection rate',
    registry=registry
)

high_amount_transactions = Counter(
    'fraud_high_amount_transactions_total',
    'High amount transactions processed',
    registry=registry
)

# Métriques de drift temps réel
drift_score_gauge = Gauge(
    'fraud_api_drift_score_current',
    'Current real-time drift score',
    registry=registry
)

drift_window_size = Gauge(
    'fraud_api_drift_window_size',
    'Current drift detection window size',
    registry=registry
)

drift_alerts_counter = Counter(
    'fraud_api_drift_alerts_total',
    'Total drift alerts triggered',
    ['severity'],
    registry=registry
)

drift_checks_counter = Counter(
    'fraud_api_drift_checks_total',
    'Total drift checks performed',
    ['result'],
    registry=registry
)

drift_reference_data_age = Gauge(
    'fraud_api_drift_reference_age_seconds',
    'Age of drift reference data in seconds',
    registry=registry
)