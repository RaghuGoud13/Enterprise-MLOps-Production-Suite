import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Union
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

class DriftDetector:
    """
    Robust Data and Model Drift Detector using Statistical Tests and PSI.
    Designed for enterprise-scale MLOps pipelines.
    """
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def calculate_ks_drift(self, reference_data: pd.Series, current_data: pd.Series) -> Dict[str, Union[float, bool]]:
        """
        Calculates Kolmogorov-Smirnov test for a single feature.
        Returns p-value and drift status.
        """
        ks_stat, p_value = stats.ks_2samp(reference_data, current_data)
        is_drift = p_value < self.threshold
        
        return {
            "p_value": float(p_value),
            "ks_stat": float(ks_stat),
            "is_drift": bool(is_drift)
        }

    def calculate_psi(self, reference_data: pd.Series, current_data: pd.Series, buckets: int = 10) -> float:
        """
        Calculates Population Stability Index (PSI) for a single feature.
        """
        def sub_psi(e_perc, a_perc):
            """Calculate PSI for a single bucket."""
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001
            return (e_perc - a_perc) * np.log(e_perc / a_perc)

        breakpoints = np.percentile(reference_data, np.arange(0, 100, 100 / buckets))
        
        expected_percents = np.histogram(reference_data, bins=np.append(breakpoints, np.inf))[0] / len(reference_data)
        actual_percents = np.histogram(current_data, bins=np.append(breakpoints, np.inf))[0] / len(current_data)

        psi_value = 0
        for i in range(len(expected_percents)):
            psi_value += sub_psi(expected_percents[i], actual_percents[i])

        return float(psi_value)

    def detect_drift_report(self, reference_df: pd.DataFrame, current_df: pd.DataFrame, column_mapping: Optional[Dict] = None) -> Dict:
        """
        Generates a comprehensive drift report using Evidently AI.
        """
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
        ])
        
        data_drift_report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
        return data_drift_report.as_dict()

    def check_features_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame, features: List[str]) -> Dict[str, Dict]:
        """
        Iterates over features and performs KS test and PSI.
        """
        results = {}
        for feature in features:
            ks_results = self.calculate_ks_drift(reference_df[feature], current_df[feature])
            psi_value = self.calculate_psi(reference_df[feature], current_df[feature])
            
            results[feature] = {
                "ks_p_value": ks_results["p_value"],
                "psi_value": psi_value,
                "drift_detected": ks_results["is_drift"] or psi_value > 0.2
            }
        return results

if __name__ == "__main__":
    # Example usage for verification
    ref = pd.Series(np.random.normal(0, 1, 1000))
    curr = pd.Series(np.random.normal(0.5, 1, 1000))
    
    detector = DriftDetector()
    print(f"KS results: {detector.calculate_ks_drift(ref, curr)}")
    print(f"PSI value: {detector.calculate_psi(ref, curr)}")
