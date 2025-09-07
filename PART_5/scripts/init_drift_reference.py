#!/usr/bin/env python3
"""
Script d'initialisation des données de référence pour la détection de drift
Utilise un échantillon du dataset original pour configurer la baseline
"""

import pandas as pd
import numpy as np
import requests
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_reference_data(csv_path: str, sample_size: int = 5000) -> np.ndarray:
    """
    Charge et échantillonne les données de référence depuis le CSV
    """
    logger.info(f"Loading data from {csv_path}")
    
    # Charger le dataset
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Prendre seulement les transactions normales (Class == 0) pour la référence
    normal_transactions = df[df['Class'] == 0]
    logger.info(f"Found {len(normal_transactions)} normal transactions")
    
    # Échantillonner aléatoirement
    if len(normal_transactions) > sample_size:
        sample = normal_transactions.sample(n=sample_size, random_state=42)
    else:
        sample = normal_transactions
    
    logger.info(f"Using {len(sample)} transactions as reference")
    
    # Colonnes de features (exclure Class)
    feature_cols = [col for col in df.columns if col != 'Class']
    
    # Retourner les features sous forme de numpy array
    return sample[feature_cols].values

def send_reference_data(api_url: str, reference_data: np.ndarray):
    """
    Envoie les données de référence à l'API
    """
    logger.info(f"Sending reference data to {api_url}")
    
    # Convertir en liste pour JSON
    data = {
        "reference_data": reference_data.tolist()
    }
    
    try:
        response = requests.post(
            f"{api_url}/v1/monitoring/drift/reference",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"✅ Reference data configured successfully: {result}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to send reference data: {e}")
        return False

def verify_drift_status(api_url: str):
    """
    Vérifie le statut de la détection de drift
    """
    try:
        response = requests.get(f"{api_url}/v1/monitoring/drift/status")
        response.raise_for_status()
        
        status = response.json()
        logger.info("📊 Drift Detection Status:")
        logger.info(f"  - Reference data available: {status['drift_status']['reference_data_available']}")
        logger.info(f"  - Window size: {status['drift_status']['window_size']}/{status['drift_status']['max_window_size']}")
        logger.info(f"  - Current drift score: {status['drift_status']['current_drift_score']:.4f}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get drift status: {e}")

def main():
    parser = argparse.ArgumentParser(description="Initialize drift detection reference data")
    parser.add_argument("--csv-path", 
                       default="../PART_4/data/creditcard.csv",
                       help="Path to the creditcard dataset CSV")
    parser.add_argument("--api-url", 
                       default="http://localhost:8000",
                       help="API base URL")
    parser.add_argument("--sample-size", 
                       type=int, 
                       default=5000,
                       help="Number of reference samples to use")
    
    args = parser.parse_args()
    
    # Vérifier que le fichier existe
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    try:
        # 1. Charger les données de référence
        reference_data = load_reference_data(str(csv_path), args.sample_size)
        
        # 2. Envoyer à l'API
        success = send_reference_data(args.api_url, reference_data)
        
        if success:
            # 3. Vérifier le statut
            verify_drift_status(args.api_url)
            logger.info("🎉 Drift detection reference data initialized successfully!")
        else:
            logger.error("❌ Failed to initialize reference data")
            
    except Exception as e:
        logger.error(f"Error during initialization: {e}")

if __name__ == "__main__":
    main()