"""
Production Model Serving API
----------------------------
Author: Senior MLOps Engineer
Description: FastAPI-based model inference service with built-in health checks 
             and Prometheus monitoring metrics.
"""

import os
import joblib
import logging
import time
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app
from pythonjsonlogger import jsonlogger

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/iris_classifier_latest.joblib")

# Logging setup
logger = logging.getLogger("api_logger")
log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

# Prometheus Metrics
PREDICTION_COUNTER = Counter("model_predictions_total", "Total model predictions")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Time spent processing prediction")

app = FastAPI(title="MLOps Production Inference Service", version="1.0.0")

# Input/Output Models
class PredictionInput(BaseModel):
    features: List[float] = Field(..., example=[5.1, 3.5, 1.4, 0.2], description="Iris features: sepal_len, sepal_width, petal_len, petal_width")

class PredictionOutput(BaseModel):
    class_id: int
    probability: float

# Global model state
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully", extra={"model_path": MODEL_PATH})
    except Exception as e:
        logger.error("Failed to load model on startup", exc_info=True)

@app.get("/health/live")
def liveness_probe():
    """Confirms the container is alive."""
    return {"status": "ok"}

@app.get("/health/ready")
def readiness_probe():
    """Confirms the model is loaded and ready to serve traffic."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "ready"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """Core inference endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    
    start_time = time.time()
    try:
        # Preprocessing & Inference
        features = [input_data.features]
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        
        # Metrics Tracking
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_COUNTER.inc()
        
        logger.info("Prediction generated", extra={
            "input": input_data.features, 
            "prediction": prediction,
            "latency": latency
        })
        
        return PredictionOutput(
            class_id=prediction,
            probability=float(probabilities[prediction])
        )
    except Exception as e:
        logger.error("Inference failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Inference processing error")

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
