import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
import logging

class TrainingOrchestrator:
    """
    High-level Training Orchestrator.
    Manages experiment tracking, artifacts, and automated model promotion.
    Supports integration with MLflow and Azure ML.
    """

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.logger = logging.getLogger(__name__)

    def start_training_run(self, model: Any, X_train: Any, y_train: Any, params: Dict[str, Any], metrics: Dict[str, Any]):
        """Runs training, logs parameters, metrics, and saves the model to MLflow."""
        with mlflow.start_run() as run:
            # Log params and metrics
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            # Log the model
            mlflow.sklearn.log_model(model, "model")
            
            run_id = run.info.run_id
            self.logger.info(f"MLflow run complete: {run_id}")
            return run_id

    def promote_model(self, run_id: str, model_name: str, stage: str = "Production"):
        """Promotes a model to a given stage in the MLflow Model Registry."""
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=stage,
            archive_existing_versions=True
        )
        self.logger.info(f"Model {model_name} promoted to {stage} (version {result.version})")

    def azure_ml_register(self, model_path: str, model_name: str, subscription_id: str, resource_group: str, workspace_name: str):
        """Registers a model to the Azure ML Workspace."""
        ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group, workspace_name
        )
        
        my_model = Model(
            path=model_path,
            type="custom_model",
            name=model_name,
            description="Model registered from Enterprise MLOps Orchestrator"
        )
        
        ml_client.models.create_or_update(my_model)
        self.logger.info(f"Model {model_name} registered in Azure ML workspace: {workspace_name}")

    def evaluate_and_promote(self, run_id: str, model_name: str, metric_name: str, threshold: float):
        """Conditional promotion based on a performance metric."""
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        metric_value = run.data.metrics.get(metric_name)
        
        if metric_value and metric_value >= threshold:
            self.logger.info(f"Metric {metric_name} ({metric_value}) meets threshold ({threshold}). Promoting...")
            self.promote_model(run_id, model_name)
        else:
            self.logger.warning(f"Metric {metric_name} ({metric_value}) below threshold. Model not promoted.")

if __name__ == "__main__":
    # Example usage for verification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    
    data = load_iris()
    X, y = data.data, data.target
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X, y)
    
    orchestrator = TrainingOrchestrator(experiment_name="Iris_Classification_Experiment")
    run_id = orchestrator.start_training_run(clf, X, y, {"n_estimators": 10}, {"accuracy": 0.95})
    # orchestrator.evaluate_and_promote(run_id, "Iris_Model", "accuracy", 0.90)
