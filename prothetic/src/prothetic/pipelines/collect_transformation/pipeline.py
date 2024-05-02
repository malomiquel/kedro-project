"""
This is a boilerplate pipeline 'collect_transformation'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import transform_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=transform_data,
            inputs="raw_daily_data",
            outputs="shaped_datas",
            name="node_merge_raw_daily_data"
        )
    ])
