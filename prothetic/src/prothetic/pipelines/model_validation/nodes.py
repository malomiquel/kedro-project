"""
This is a boilerplate pipeline 'model_validation'
generated using Kedro 0.19.5
"""

import pandas as pd
from keras import Model
from mlflow import log_metrics

def evaluate_model(model: Model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """
    Évalue le modèle sur les données de test et enregistre les métriques avec MLflow.
    
    Args:
    - model (Model): Le modèle de réseau de neurones à évaluer.
    - X_test (pd.DataFrame): Les données de test pour les features.
    - y_test (pd.DataFrame): Les données de test pour les labels.

    Returns:
    - evaluation_metrics (dict): Un dictionnaire contenant les métriques d'évaluation du modèle.
    """
    # Évalue le modèle sur les données de test
    results = model.evaluate(X_test, y_test, verbose=1)
    
    # Crée un dictionnaire des métriques d'évaluation
    evaluation_metrics = {
        "loss": results[0],  # La perte du modèle
        "accuracy": results[1]  # L'exactitude du modèle
    }
    
    # Enregistre les métriques d'évaluation avec MLflow
    log_metrics(evaluation_metrics)
    
    return evaluation_metrics