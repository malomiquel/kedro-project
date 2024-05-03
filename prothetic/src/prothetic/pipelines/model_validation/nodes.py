"""
This is a boilerplate pipeline 'model_validation'
generated using Kedro 0.19.5
"""
import pandas as pd
from keras import Model
from mlflow import log_metrics

def evaluate_model(model: Model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    results = model.evaluate(X_test, y_test, verbose=1)
    
    evaluation_metrics = {
        "loss": results[0],
        "accuracy": results[1]
    }
    
    log_metrics(evaluation_metrics)
    
    return evaluation_metrics
