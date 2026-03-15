from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any, Optional

class FeatureStoreInterface(ABC):
    """
    Abstract Base Class for Feature Store connectivity.
    Ensures modularity and support for multiple providers (Hopsworks, Azure ML, etc.).
    """

    @abstractmethod
    def get_features(self, feature_names: List[str], entity_id: str) -> pd.DataFrame:
        """Fetch online features for a specific entity."""
        pass

    @abstractmethod
    def get_training_dataset(self, feature_names: List[str], start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch historical features for offline training."""
        pass

    @abstractmethod
    def push_features(self, data: pd.DataFrame, feature_group_name: str):
        """Upload features to the feature store."""
        pass

class AzureMLFeatureStore(FeatureStoreInterface):
    """
    Azure ML Feature Store Implementation.
    Requires azure-ai-ml client and workspace configuration.
    """
    def __init__(self, workspace_name: str, resource_group: str, subscription_id: str):
        # In a real implementation, initialize the MLClient here
        self.workspace_name = workspace_name
        print(f"Connected to Azure ML Feature Store: {workspace_name}")

    def get_features(self, feature_names: List[str], entity_id: str) -> pd.DataFrame:
        print(f"Fetching online features from Azure ML for: {entity_id}")
        return pd.DataFrame() # Stub

    def get_training_dataset(self, feature_names: List[str], start_time: str, end_time: str) -> pd.DataFrame:
        print(f"Fetching historical data for training from Azure ML")
        return pd.DataFrame() # Stub

    def push_features(self, data: pd.DataFrame, feature_group_name: str):
        print(f"Pushing features to Azure ML Feature Store: {feature_group_name}")

class HopsworksFeatureStore(FeatureStoreInterface):
    """
    Hopsworks Feature Store Implementation.
    Requires hopsworks python library.
    """
    def __init__(self, api_key: str, project_name: str):
        # In a real implementation, initialize hopsworks.login()
        self.project_name = project_name
        print(f"Connected to Hopsworks: {project_name}")

    def get_features(self, feature_names: List[str], entity_id: str) -> pd.DataFrame:
        print(f"Fetching online features from Hopsworks for: {entity_id}")
        return pd.DataFrame() # Stub

    def get_training_dataset(self, feature_names: List[str], start_time: str, end_time: str) -> pd.DataFrame:
        print(f"Fetching training data from Hopsworks")
        return pd.DataFrame() # Stub

    def push_features(self, data: pd.DataFrame, feature_group_name: str):
        print(f"Pushing features to Hopsworks: {feature_group_name}")

class FeatureStoreFactory:
    """
    Factory to instantiate the appropriate feature store provider.
    """
    @staticmethod
    def get_provider(provider_type: str, **kwargs) -> FeatureStoreInterface:
        if provider_type.lower() == "azure":
            return AzureMLFeatureStore(**kwargs)
        elif provider_type.lower() == "hopsworks":
            return HopsworksFeatureStore(**kwargs)
        else:
            raise ValueError(f"Unsupported Feature Store provider: {provider_type}")
