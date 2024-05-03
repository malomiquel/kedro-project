"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import create_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_model,
            inputs="shaped_datas",
            outputs="model",
            name="node_create_model"
        )
    ])
