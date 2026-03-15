# Terraform script stub for Enterprise MLOps Infrastructure on Azure
# Configures AKS (Azure Kubernetes Service) and ACR (Azure Container Registry)

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "resource_group_name" {
  description = "Resource Group Name"
  default     = "mlops-enterprise-rg"
}

variable "location" {
  description = "Azure Region"
  default     = "East US"
}

variable "cluster_name" {
  description = "AKS Cluster Name"
  default     = "mlops-production-aks"
}

variable "acr_name" {
  description = "Azure Container Registry Name"
  default     = "mlopsacrregistry"
}

resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_container_registry" "acr" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Standard"
  admin_enabled       = true
}

resource "azurerm_kubernetes_cluster" "aks" {
  name                = var.cluster_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = "mlopsaks"

  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_DS2_v2"
  }

  identity {
    type = "SystemAssigned"
  }

  tags = {
    Environment = "Production"
    Project     = "MLOps-Lifecycle-Engine"
  }
}

# Output the AKS Kubeconfig
output "kube_config" {
  value     = azurerm_kubernetes_cluster.aks.kube_config_raw
  sensitive = true
}

output "acr_login_server" {
  value = azurerm_container_registry.acr.login_server
}
