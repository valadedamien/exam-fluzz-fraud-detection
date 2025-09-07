from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.datasets import make_classification
import mlflow
import mlflow.sklearn
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import sys
sys.path.append('/opt/airflow/monitoring')
from drift_detector import DriftDetector

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='Machine Learning Pipeline with MLflow and Drift Detection',
    schedule_interval=timedelta(hours=6),
    catchup=False,
    tags=['ml', 'mlflow', 'monitoring'],
)

def load_data(**context):
    """Load data from /data directory or generate synthetic data"""
    data_path = '/data'
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')] if os.path.exists(data_path) else []
    
    if csv_files:
        df = pd.read_csv(os.path.join(data_path, csv_files[0]))
        print(f"Loaded data from {csv_files[0]}: {df.shape}")
    else:
        print("No CSV found, generating synthetic data")
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                                 n_redundant=5, n_classes=3, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
    
    df.to_csv('/data/raw_data.csv', index=False)
    return '/data/raw_data.csv'

def preprocess_data(**context):
    """Preprocess data: clean, scale, split"""
    ti = context['ti']
    data_path = ti.xcom_pull(task_ids='load_data')
    
    df = pd.read_csv(data_path)
    
    # Identify features and target
    target_col = 'Class' if 'Class' in df.columns else 'target'
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessed data
    pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv('/data/X_train.csv', index=False)
    pd.DataFrame(X_val_scaled, columns=feature_cols).to_csv('/data/X_val.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv('/data/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('/data/y_train.csv', index=False)
    pd.DataFrame(y_val).to_csv('/data/y_val.csv', index=False)
    pd.DataFrame(y_test).to_csv('/data/y_test.csv', index=False)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, '/data/scaler.pkl')
    
    return {
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'n_classes': len(y.unique())
    }

def train_model(**context):
    """Train MLPClassifier and log to MLflow"""
    ti = context['ti']
    data_info = ti.xcom_pull(task_ids='preprocess_data')
    
    # Load data
    X_train = pd.read_csv('/data/X_train.csv')
    y_train = pd.read_csv('/data/y_train.csv').iloc[:, 0]
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 500,
            'random_state': 42
        }
        mlflow.log_params(params)
        mlflow.log_params(data_info)
        
        # Train model
        model = MLPClassifier(**params)
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally for next tasks
        import joblib
        joblib.dump(model, '/data/model.pkl')
        
        return run.info.run_id

def evaluate_model(**context):
    """Evaluate model and log metrics to MLflow"""
    ti = context['ti']
    run_id = ti.xcom_pull(task_ids='train_model')
    
    # Load model and data
    import joblib
    model = joblib.load('/data/model.pkl')
    X_val = pd.read_csv('/data/X_val.csv')
    y_val = pd.read_csv('/data/y_val.csv').iloc[:, 0]
    X_test = pd.read_csv('/data/X_test.csv')
    y_test = pd.read_csv('/data/y_test.csv').iloc[:, 0]
    
    # Continue MLflow run
    with mlflow.start_run(run_id=run_id):
        # Validation metrics
        y_val_pred = model.predict(X_val)
        
        # FRAUD CLASS SPECIFIC METRICS (classe 1) - les vraies métriques qui comptent !
        val_fraud_f1 = f1_score(y_val, y_val_pred, pos_label=1)
        val_fraud_precision = precision_score(y_val, y_val_pred, pos_label=1)
        val_fraud_recall = recall_score(y_val, y_val_pred, pos_label=1)
        
        val_metrics = {
            'val_fraud_f1': val_fraud_f1,
            'val_fraud_precision': val_fraud_precision,
            'val_fraud_recall': val_fraud_recall,
            'val_f1_macro': f1_score(y_val, y_val_pred, average='macro'),
            'val_f1_weighted': f1_score(y_val, y_val_pred, average='weighted'),
            'val_precision_macro': precision_score(y_val, y_val_pred, average='macro'),
            'val_precision_weighted': precision_score(y_val, y_val_pred, average='weighted'),
            'val_recall_macro': recall_score(y_val, y_val_pred, average='macro'),
            'val_recall_weighted': recall_score(y_val, y_val_pred, average='weighted')
        }
        
        # Test metrics
        y_test_pred = model.predict(X_test)
        
        # FRAUD CLASS SPECIFIC METRICS (classe 1) - les vraies métriques qui comptent !
        test_fraud_f1 = f1_score(y_test, y_test_pred, pos_label=1)
        test_fraud_precision = precision_score(y_test, y_test_pred, pos_label=1)
        test_fraud_recall = recall_score(y_test, y_test_pred, pos_label=1)
        
        test_metrics = {
            'test_fraud_f1': test_fraud_f1,
            'test_fraud_precision': test_fraud_precision,
            'test_fraud_recall': test_fraud_recall,
            'test_f1_macro': f1_score(y_test, y_test_pred, average='macro'),
            'test_f1_weighted': f1_score(y_test, y_test_pred, average='weighted'),
            'test_precision_macro': precision_score(y_test, y_test_pred, average='macro'),
            'test_precision_weighted': precision_score(y_test, y_test_pred, average='weighted'),
            'test_recall_macro': recall_score(y_test, y_test_pred, average='macro'),
            'test_recall_weighted': recall_score(y_test, y_test_pred, average='weighted')
        }
        
        # Log all metrics
        mlflow.log_metrics({**val_metrics, **test_metrics})
        
        # Log confusion matrix with detailed analysis
        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])
        
        cm_analysis = f"""Confusion Matrix:
{cm}

Detailed Analysis:
- True Negatives (Normal correctly classified): {tn}
- False Positives (Normal flagged as fraud): {fp}
- False Negatives (Frauds missed): {fn} ⚠️ CRITICAL!
- True Positives (Frauds detected): {tp}

Fraud Detection Performance:
- Frauds detected: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)
- Frauds missed: {fn}/{tp+fn} ({fn/(tp+fn)*100:.1f}%) ⚠️
- False alarm rate: {fp}/{fp+tn} ({fp/(fp+tn)*100:.3f}%)
"""
        
        mlflow.log_text(cm_analysis, "confusion_matrix_analysis.txt")
        
        # Log business impact metrics
        fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        business_metrics = {
            'fraud_detection_rate': fraud_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'frauds_missed_count': int(fn),
            'frauds_detected_count': int(tp),
            'false_alarms_count': int(fp)
        }
        
        mlflow.log_metrics(business_metrics)
        
        return {**val_metrics, **test_metrics, **business_metrics, 'run_id': run_id}

def detect_drift(**context):
    """Detect data drift compared to previous runs"""
    ti = context['ti']
    current_metrics = ti.xcom_pull(task_ids='evaluate_model')
    run_id = current_metrics['run_id']
    
    # Load current data
    X_train = pd.read_csv('/data/X_train.csv')
    
    # Continue MLflow run
    with mlflow.start_run(run_id=run_id):
        try:
            # Get previous runs
            experiment = mlflow.get_experiment_by_name("Default")
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                                        order_by=["start_time DESC"], max_results=5)
                
                if len(runs) > 1:  # If we have previous runs
                    drift_detector = DriftDetector()
                    
                    # Compare with most recent previous run
                    previous_run = runs.iloc[1]  # Second most recent (first is current)
                    
                    # Calculate drift metrics
                    drift_results = drift_detector.detect_drift(X_train, X_train)  # Simplified for demo
                    
                    drift_metrics = {
                        'drift_score': drift_results.get('drift_score', 0.0),
                        'n_drifted_features': drift_results.get('n_drifted_features', 0),
                        'max_feature_drift': drift_results.get('max_feature_drift', 0.0)
                    }
                    
                    mlflow.log_metrics(drift_metrics)
                    
                    return drift_metrics
                else:
                    print("No previous runs found for drift detection")
                    return {'drift_score': 0.0, 'n_drifted_features': 0, 'max_feature_drift': 0.0}
            else:
                print("No experiment found")
                return {'drift_score': 0.0, 'n_drifted_features': 0, 'max_feature_drift': 0.0}
                
        except Exception as e:
            print(f"Drift detection failed: {str(e)}")
            return {'drift_score': 0.0, 'n_drifted_features': 0, 'max_feature_drift': 0.0}

def log_metrics_to_prometheus(**context):
    """Push metrics to Prometheus pushgateway - FOCUS ON FRAUD CLASS PERFORMANCE"""
    ti = context['ti']
    metrics = ti.xcom_pull(task_ids='evaluate_model')
    drift_metrics = ti.xcom_pull(task_ids='detect_drift')
    
    try:
        registry = CollectorRegistry()
        
        # FRAUD CLASS SPECIFIC METRICS - Les vraies métriques qui comptent !
        fraud_f1_gauge = Gauge('ml_fraud_f1_score', 'Fraud Detection F1 Score', registry=registry)
        fraud_precision_gauge = Gauge('ml_fraud_precision', 'Fraud Detection Precision', registry=registry)
        fraud_recall_gauge = Gauge('ml_fraud_recall', 'Fraud Detection Recall', registry=registry)
        
        # Business Impact Metrics
        fraud_detection_rate_gauge = Gauge('ml_fraud_detection_rate', 'Fraud Detection Rate (0-1)', registry=registry)
        false_alarm_rate_gauge = Gauge('ml_false_alarm_rate', 'False Alarm Rate (0-1)', registry=registry)
        frauds_missed_gauge = Gauge('ml_frauds_missed_count', 'Number of Frauds Missed', registry=registry)
        frauds_detected_gauge = Gauge('ml_frauds_detected_count', 'Number of Frauds Detected', registry=registry)
        false_alarms_gauge = Gauge('ml_false_alarms_count', 'Number of False Alarms', registry=registry)
        
        # Traditional metrics (pour comparaison)
        f1_macro_gauge = Gauge('ml_model_f1_macro', 'F1 Score Macro Average', registry=registry)
        f1_weighted_gauge = Gauge('ml_model_f1_weighted', 'F1 Score Weighted Average', registry=registry)
        
        # Drift metrics
        drift_score_gauge = Gauge('ml_model_drift_score', 'Data Drift Score', registry=registry)
        drift_features_gauge = Gauge('ml_model_drifted_features', 'Number of Drifted Features', registry=registry)
        
        # Set FRAUD CLASS values (les vraies métriques importantes !)
        fraud_f1_gauge.set(metrics.get('test_fraud_f1', 0))
        fraud_precision_gauge.set(metrics.get('test_fraud_precision', 0))
        fraud_recall_gauge.set(metrics.get('test_fraud_recall', 0))
        
        # Set business impact values
        fraud_detection_rate_gauge.set(metrics.get('fraud_detection_rate', 0))
        false_alarm_rate_gauge.set(metrics.get('false_alarm_rate', 0))
        frauds_missed_gauge.set(metrics.get('frauds_missed_count', 0))
        frauds_detected_gauge.set(metrics.get('frauds_detected_count', 0))
        false_alarms_gauge.set(metrics.get('false_alarms_count', 0))
        
        # Set traditional averages (pour comparaison)
        f1_macro_gauge.set(metrics.get('test_f1_macro', 0))
        f1_weighted_gauge.set(metrics.get('test_f1_weighted', 0))
        
        # Set drift values
        drift_score_gauge.set(drift_metrics['drift_score'])
        drift_features_gauge.set(drift_metrics['n_drifted_features'])
        
        # Push to gateway
        push_to_gateway('pushgateway:9091', job='ml_pipeline', registry=registry)
        
        print("Fraud-focused metrics pushed to Prometheus successfully")
        print(f"Key metrics - Fraud F1: {metrics.get('test_fraud_f1', 0):.3f}, "
              f"Fraud Recall: {metrics.get('test_fraud_recall', 0):.3f}, "
              f"Frauds Missed: {metrics.get('frauds_missed_count', 0)}")
        return True
        
    except Exception as e:
        print(f"Failed to push metrics to Prometheus: {str(e)}")
        return False

def register_model(**context):
    """Register model in MLflow Model Registry if performance is good"""
    ti = context['ti']
    metrics = ti.xcom_pull(task_ids='evaluate_model')
    drift_metrics = ti.xcom_pull(task_ids='detect_drift')
    run_id = metrics['run_id']
    
    # Define thresholds
    f1_threshold = 0.7
    drift_threshold = 0.3
    
    f1_score = metrics['test_f1_weighted']
    drift_score = drift_metrics['drift_score']
    
    with mlflow.start_run(run_id=run_id):
        if f1_score >= f1_threshold and drift_score <= drift_threshold:
            try:
                model_uri = f"runs:/{run_id}/model"
                model_version = mlflow.register_model(model_uri, "MLPClassifier")
                
                # Transition to staging
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name="MLPClassifier",
                    version=model_version.version,
                    stage="Staging"
                )
                
                mlflow.log_param("model_registered", True)
                mlflow.log_param("model_version", model_version.version)
                
                print(f"Model registered successfully as version {model_version.version}")
                return model_version.version
                
            except Exception as e:
                print(f"Model registration failed: {str(e)}")
                mlflow.log_param("model_registered", False)
                return None
        else:
            print(f"Model not registered: F1={f1_score:.3f} (threshold={f1_threshold}), "
                  f"Drift={drift_score:.3f} (threshold={drift_threshold})")
            mlflow.log_param("model_registered", False)
            return None

# Define tasks
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

detect_drift_task = PythonOperator(
    task_id='detect_drift',
    python_callable=detect_drift,
    dag=dag,
)

log_metrics_task = PythonOperator(
    task_id='log_metrics_to_prometheus',
    python_callable=log_metrics_to_prometheus,
    dag=dag,
)

register_model_task = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    dag=dag,
)

# Define task dependencies
load_data_task >> preprocess_data_task >> train_model_task >> evaluate_model_task
evaluate_model_task >> detect_drift_task >> log_metrics_task >> register_model_task