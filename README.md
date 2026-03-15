# Enterprise MLOps Production Suite

A professional, production-ready MLOps framework for end-to-end model development, validation, and serving. This repository implements a modular scikit-learn training pipeline and a FastAPI-based inference service, designed for high-availability Kubernetes deployments.

## 🚀 Key Features

- **Modular Training Pipeline**: Built with clean abstractions, structured logging, and automated validation thresholds.
- **Model Versioning**: Automated artifact versioning and "latest" tagging for seamless deployment.
- **Production Inference API**: FastAPI service with Prometheus monitoring, structured JSON logging, and Kubernetes health probes.
- **Infrastructure as Code**: Multi-stage optimized Docker builds and K8s manifests for HPA-enabled deployments.
- **Robust CI/CD Support**: Designed for Azure DevOps, GitHub Actions, or GitLab CI/CD integration.

## 🏗️ Project Structure

```text
.
├── artifacts/             # Local model storage (git-ignored)
├── data/                  # Local datasets (git-ignored)
├── infrastructure/        # DevOps & Deployment configs
│   ├── Dockerfile         # Multi-stage optimized build
│   └── k8s-deployment.yml # Kubernetes manifest (HPA, Probes)
├── src/
│   ├── api/               # FastAPI Inference service
│   └── training/          # Training pipeline & validation logic
├── tests/                 # Pytest suite for training/API
├── requirements.txt       # Production dependencies
└── .gitignore             # Standard Python/ML gitignore
```

## 🛠️ Usage

### 1. Training the Model
To execute the E2E training lifecycle, run:
```bash
export ARTIFACT_DIR=artifacts
python src/training/pipeline.py
```
This will:
- Load and split the Iris dataset.
- Train a Random Forest classifier.
- Validate the model (fails if accuracy < 0.90).
- Save versioned and `latest` model artifacts.

### 2. Running the Inference Service
Once a model artifact is available:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
- **Inference**: POST to `/predict` with `{"features": [5.1, 3.5, 1.4, 0.2]}`.
- **Metrics**: GET `/metrics` for Prometheus data.
- **Health**: GET `/health/live` and `/health/ready`.

### 3. Testing
Run the comprehensive test suite:
```bash
pytest tests/
```

## ☸️ Kubernetes Deployment

Deploy the model service to a Kubernetes cluster (e.g., AKS):
```bash
kubectl apply -f infrastructure/k8s-deployment.yml
```
Features included in the manifest:
- **Rolling Updates**: Zero-downtime deployments.
- **HPA**: Horizontal Pod Autoscaling (3-10 replicas based on CPU).
- **Liveness/Readiness Probes**: Automated self-healing.
- **Prometheus Scraping**: Metrics automatically collected by Prometheus.

## 🔄 CI/CD Strategy (Azure DevOps Style)

The project is structured to support a 3-stage pipeline:
1. **CI Pipeline (Build)**:
   - Run `pytest` on code changes.
   - Execute `src/training/pipeline.py` to generate artifacts.
   - Build and push Docker image to Azure Container Registry (ACR).
2. **Release Pipeline (Staging)**:
   - Deploy image to AKS Staging namespace.
   - Run integration tests against the live endpoint.
3. **Release Pipeline (Production)**:
   - Manual approval gate.
   - Deploy to AKS Production namespace with blue/green or canary strategy.

---
**Author**: Senior MLOps Research Engineer
**License**: MIT
