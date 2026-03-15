# Stage 1: Build & Dependency Resolution
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime Environment
FROM python:3.9-slim

# Security: Create non-root user
RUN groupadd -r mlops && useradd -r -g mlops mluser
WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /home/mluser/.local
ENV PATH=/home/mluser/.local/bin:$PATH

# Copy application source code
COPY src/api/ /app/api/
COPY artifacts/ /app/artifacts/

# Set ownership to non-root user
RUN chown -R mluser:mlops /app
USER mluser

# Environment variables
ENV MODEL_PATH=/app/artifacts/iris_classifier_latest.joblib
ENV PYTHONUNBUFFERED=1

# Expose FastAPI and Metrics ports
EXPOSE 8000

# Health checks for Kubernetes
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health/live || exit 1

ENTRYPOINT ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
