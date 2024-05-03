"""
This is a boilerplate pipeline 'collect_transformation'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import transform_data, split_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=transform_data,
            inputs="raw_daily_data",
            outputs="shaped_datas",
            name="node_merge_raw_daily_data"
        ),
        node(
            func=split_dataset,
            inputs="shaped_datas",
            outputs=["x_train", "x_test", "y_train", "y_test"],
            name="node_split_transform_daily_data"
        )
    ])
