"""Project pipelines."""
from typing import Dict
# from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from prothetic.pipelines import (
    collect_transformation,
    model_training,
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.
    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    my_pipeline = collect_transformation.create_pipeline()
    my_pipeline_2 = model_training.create_pipeline()
    # Return statement indicates the default sequence of modular pipeline
    return {
        "collect_transformation": my_pipeline ,
        "model_training": my_pipeline_2 ,
        "__default__": my_pipeline + my_pipeline_2
    }