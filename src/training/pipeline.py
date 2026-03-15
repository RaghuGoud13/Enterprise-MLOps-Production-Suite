"""
Modular MLOps Training Pipeline
--------------------------------
Author: Senior MLOps Engineer
Description: Modular scikit-learn training pipeline with built-in validation, 
             artifact versioning, and enterprise logging.
"""

import os
import joblib
import logging
import datetime
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pythonjsonlogger import jsonlogger

# Configuration
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_NAME = "iris_classifier"

# Logging setup
logger = logging.getLogger()
log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

class TrainingPipeline:
    def __init__(self, model_params: Dict[str, Any] = None):
        self.model_params = model_params or {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
        self.model = RandomForestClassifier(**self.model_params)
        os.makedirs(ARTIFACT_DIR, exist_ok=True)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads and prepares dataset for training."""
        logger.info("Loading dataset...")
        data = load_iris()
        X, y = data.data, data.target
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def validate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluates model performance against business-level thresholds."""
        logger.info("Validating model...")
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        metrics = {
            "accuracy": accuracy,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        logger.info("Validation metrics", extra=metrics)
        
        if accuracy < 0.90:
            raise ValueError(f"Model accuracy ({accuracy:.2f}) is below deployment threshold (0.90)")
        
        return metrics

    def train(self) -> str:
        """Executes full E2E training lifecycle."""
        try:
            X_train, X_test, y_train, y_test = self.load_data()
            
            logger.info("Starting model training", extra={"params": self.model_params})
            self.model.fit(X_train, y_train)
            
            # Validation Step
            self.validate_model(X_test, y_test)
            
            # Versioning & Export
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_name = f"{MODEL_NAME}_{timestamp}.joblib"
            artifact_path = os.path.join(ARTIFACT_DIR, versioned_name)
            latest_path = os.path.join(ARTIFACT_DIR, f"{MODEL_NAME}_latest.joblib")
            
            joblib.dump(self.model, artifact_path)
            joblib.dump(self.model, latest_path) # Symlink equivalent for latest
            
            logger.info("Training complete", extra={"artifact_path": artifact_path})
            return artifact_path
            
        except Exception as e:
            logger.error("Training failed", exc_info=True)
            raise

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.train()
