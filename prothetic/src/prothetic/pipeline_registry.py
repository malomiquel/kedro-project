"""Project pipelines."""
from typing import Dict
# from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from prothetic.pipelines import (
    collect_transformation,
    model_training,
    model_validation,
    model_deployment,
    model_prediction
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.
    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    collect_transformation_pipeline = collect_transformation.create_pipeline()
    model_training_pipeline = model_training.create_pipeline()
    model_validation_pipeline = model_validation.create_pipeline()
    model_deployment_pipeline = model_deployment.create_pipeline()
    model_prediction_pipeline = model_prediction.create_pipeline()
    # Return statement indicates the default sequence of modular pipeline
    return {
        "collect_transformation": collect_transformation_pipeline,
        "model_training": model_training_pipeline,
        "model_validation": model_validation_pipeline,
        "model_deployment": model_deployment_pipeline,
        "model_prediction": model_prediction_pipeline,
        "__default__": collect_transformation_pipeline + model_training_pipeline + model_validation_pipeline + model_deployment_pipeline
    }
