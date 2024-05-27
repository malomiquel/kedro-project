"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=["x_train", "y_train", "x_test", "y_test"],
            outputs="model_trained",
            name="node_train_model"
        )
    ])
