import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DriftDetector:
    """
    Data drift detection using statistical tests
    """
    
    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level
        self.drift_threshold = 0.1  # PSI threshold for drift
        
    def detect_drift(self, reference_data, current_data, categorical_cols=None):
        """
        Detect drift between reference and current datasets
        
        Args:
            reference_data: pandas DataFrame - reference (baseline) dataset
            current_data: pandas DataFrame - current dataset to compare
            categorical_cols: list - names of categorical columns
            
        Returns:
            dict: drift detection results
        """
        if isinstance(reference_data, pd.DataFrame) and isinstance(current_data, pd.DataFrame):
            reference_data = reference_data.values
            current_data = current_data.values
            
        # Ensure both datasets have same number of features
        if reference_data.shape[1] != current_data.shape[1]:
            raise ValueError("Reference and current data must have same number of features")
            
        n_features = reference_data.shape[1]
        drift_results = {}
        feature_drift_scores = []
        drifted_features = []
        
        for feature_idx in range(n_features):
            ref_feature = reference_data[:, feature_idx]
            curr_feature = current_data[:, feature_idx]
            
            # Remove NaN values
            ref_feature = ref_feature[~np.isnan(ref_feature)]
            curr_feature = curr_feature[~np.isnan(curr_feature)]
            
            if len(ref_feature) == 0 or len(curr_feature) == 0:
                continue
                
            # Detect if feature is categorical or continuous
            ref_unique = len(np.unique(ref_feature))
            curr_unique = len(np.unique(curr_feature))
            
            # If less than 10 unique values, treat as categorical
            if ref_unique < 10 and curr_unique < 10:
                drift_score = self._categorical_drift(ref_feature, curr_feature)
            else:
                drift_score = self._numerical_drift(ref_feature, curr_feature)
            
            feature_drift_scores.append(drift_score)
            
            if drift_score > self.drift_threshold:
                drifted_features.append(feature_idx)
        
        # Calculate overall drift metrics
        overall_drift_score = np.mean(feature_drift_scores) if feature_drift_scores else 0.0
        max_drift_score = np.max(feature_drift_scores) if feature_drift_scores else 0.0
        
        # Population Stability Index (PSI) for overall dataset
        psi_score = self._calculate_psi(reference_data, current_data)
        
        drift_results = {
            'drift_score': overall_drift_score,
            'max_feature_drift': max_drift_score,
            'n_drifted_features': len(drifted_features),
            'drifted_features': drifted_features,
            'feature_drift_scores': feature_drift_scores,
            'psi_score': psi_score,
            'is_drift_detected': overall_drift_score > self.drift_threshold or len(drifted_features) > 0
        }
        
        return drift_results
    
    def _numerical_drift(self, reference, current):
        """
        Detect drift in numerical features using KS test and PSI
        """
        try:
            # Kolmogorov-Smirnov test
            ks_stat, ks_p_value = ks_2samp(reference, current)
            
            # Population Stability Index
            psi = self._calculate_feature_psi(reference, current)
            
            # Combined drift score (higher = more drift)
            drift_score = min(1.0, ks_stat + psi)
            
            return drift_score
            
        except Exception as e:
            print(f"Error in numerical drift detection: {e}")
            return 0.0
    
    def _categorical_drift(self, reference, current):
        """
        Detect drift in categorical features using chi-square test
        """
        try:
            # Get unique categories from both datasets
            all_categories = np.unique(np.concatenate([reference, current]))
            
            # Count frequencies
            ref_counts = pd.Series(reference).value_counts().reindex(all_categories, fill_value=0)
            curr_counts = pd.Series(current).value_counts().reindex(all_categories, fill_value=0)
            
            # Avoid zero counts for chi-square test
            ref_counts = ref_counts + 1
            curr_counts = curr_counts + 1
            
            # Create contingency table
            contingency_table = np.array([ref_counts.values, curr_counts.values])
            
            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate drift score based on chi-square statistic
            # Normalize by degrees of freedom
            drift_score = min(1.0, chi2 / (len(all_categories) - 1) / 100)
            
            return drift_score
            
        except Exception as e:
            print(f"Error in categorical drift detection: {e}")
            return 0.0
    
    def _calculate_feature_psi(self, reference, current, buckets=10):
        """
        Calculate Population Stability Index for a single feature
        """
        try:
            # Create buckets based on reference data quantiles
            if len(np.unique(reference)) < buckets:
                # If not enough unique values, use unique values as buckets
                bucket_edges = np.unique(reference)
                bucket_edges = np.append(bucket_edges, bucket_edges[-1] + 1)
            else:
                bucket_edges = np.quantile(reference, np.linspace(0, 1, buckets + 1))
                bucket_edges[-1] = bucket_edges[-1] + 1  # Extend last bucket
            
            # Ensure bucket edges are unique and sorted
            bucket_edges = np.unique(bucket_edges)
            
            if len(bucket_edges) <= 1:
                return 0.0
                
            # Calculate frequencies for each bucket
            ref_freq, _ = np.histogram(reference, bins=bucket_edges)
            curr_freq, _ = np.histogram(current, bins=bucket_edges)
            
            # Convert to proportions
            ref_prop = ref_freq / len(reference)
            curr_prop = curr_freq / len(current)
            
            # Avoid zero proportions
            ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
            curr_prop = np.where(curr_prop == 0, 0.0001, curr_prop)
            
            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
            
            return abs(psi)
            
        except Exception as e:
            print(f"Error in PSI calculation: {e}")
            return 0.0
    
    def _calculate_psi(self, reference_data, current_data, buckets=10):
        """
        Calculate overall PSI for the dataset
        """
        try:
            # Flatten arrays if multidimensional
            if reference_data.ndim > 1:
                reference_flat = reference_data.flatten()
                current_flat = current_data.flatten()
            else:
                reference_flat = reference_data
                current_flat = current_data
            
            # Remove NaN values
            reference_flat = reference_flat[~np.isnan(reference_flat)]
            current_flat = current_flat[~np.isnan(current_flat)]
            
            if len(reference_flat) == 0 or len(current_flat) == 0:
                return 0.0
                
            return self._calculate_feature_psi(reference_flat, current_flat, buckets)
            
        except Exception as e:
            print(f"Error in overall PSI calculation: {e}")
            return 0.0
    
    def interpret_drift(self, drift_results):
        """
        Interpret drift detection results
        """
        interpretation = {
            'status': 'No Drift',
            'severity': 'Low',
            'recommendations': []
        }
        
        drift_score = drift_results.get('drift_score', 0)
        n_drifted_features = drift_results.get('n_drifted_features', 0)
        psi_score = drift_results.get('psi_score', 0)
        
        if drift_score > 0.3 or n_drifted_features > 5 or psi_score > 0.25:
            interpretation['status'] = 'Significant Drift'
            interpretation['severity'] = 'High'
            interpretation['recommendations'] = [
                'Retrain model immediately',
                'Investigate data pipeline changes',
                'Update feature engineering'
            ]
        elif drift_score > 0.15 or n_drifted_features > 2 or psi_score > 0.1:
            interpretation['status'] = 'Moderate Drift'
            interpretation['severity'] = 'Medium'
            interpretation['recommendations'] = [
                'Monitor closely',
                'Consider retraining soon',
                'Analyze drifted features'
            ]
        elif drift_score > 0.05 or n_drifted_features > 0:
            interpretation['status'] = 'Minor Drift'
            interpretation['severity'] = 'Low'
            interpretation['recommendations'] = [
                'Continue monitoring',
                'Document changes'
            ]
        
        return interpretation