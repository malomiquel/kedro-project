"""
This is a boilerplate pipeline 'model_prediction'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import transform_data, predict_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=transform_data,
            inputs="raw_data_to_predict",
            outputs="data_to_predict",
            name="node_transform_data_to_predict"
        ),
        node(
            func=predict_model,
            inputs=["data_to_predict", "model_final"],
            outputs="data_predicted",
            name="node_predict_model"
        )
    ])
