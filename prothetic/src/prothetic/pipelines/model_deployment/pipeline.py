"""
This is a boilerplate pipeline 'model_deployment'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import choose_best_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=choose_best_model,
            inputs=["x_test", "y_test", "model_trained", "params:model_final_path"],
            outputs="model_final",
            name="node_choose_best_model"
        )
    ])