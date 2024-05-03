"""
This is a boilerplate pipeline 'model_validation'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            evaluate_model,
            inputs=["model_trained", "x_test", "y_test"],
            outputs=None,
            name="evaluate_model_node"
        )
    ])
