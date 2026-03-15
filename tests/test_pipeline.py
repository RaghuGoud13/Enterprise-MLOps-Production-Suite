"""
Comprehensive Test Suite for MLOps Pipeline
-------------------------------------------
Author: Senior MLOps Engineer
Description: Unit and integration tests for model training and serving logic.
"""

import os
import pytest
import numpy as np
from src.training.pipeline import TrainingPipeline
from src.api.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

# Pipeline Tests
def test_pipeline_initialization():
    """Validates that the training pipeline initializes with correct defaults."""
    pipeline = TrainingPipeline()
    assert pipeline.model_params["n_estimators"] == 100
    assert os.path.exists("artifacts")

def test_data_loading():
    """Ensures data loading returns expected shapes."""
    pipeline = TrainingPipeline()
    X_train, X_test, y_train, y_test = pipeline.load_data()
    assert X_train.shape[1] == 4
    assert len(np.unique(y_train)) == 3

@pytest.mark.skipif(not os.path.exists("artifacts/iris_classifier_latest.joblib"), reason="Model not trained yet")
def test_model_validation():
    """Tests model validation against the pre-defined threshold."""
    pipeline = TrainingPipeline()
    _, X_test, _, y_test = pipeline.load_data()
    
    # Mocking trained model for validation test if needed
    pipeline.train() 
    metrics = pipeline.validate_model(X_test, y_test)
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0.90

# API Tests
def test_health_endpoints():
    """Validates FastAPI health and readiness checks."""
    response_live = client.get("/health/live")
    assert response_live.status_code == 200
    assert response_live.json() == {"status": "ok"}

def test_prediction_input_validation():
    """Ensures API rejects malformed inference requests."""
    # Test with too few features
    response = client.post("/predict", json={"features": [5.1, 3.5]})
    assert response.status_code == 422 # Pydantic validation error

@pytest.mark.skipif(not os.path.exists("artifacts/iris_classifier_latest.joblib"), reason="Model not trained yet")
def test_inference_logic():
    """Integration test for full inference lifecycle."""
    test_input = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=test_input)
    
    assert response.status_code == 200
    data = response.json()
    assert "class_id" in data
    assert "probability" in data
    assert 0 <= data["class_id"] <= 2
    assert 0.0 <= data["probability"] <= 1.0
