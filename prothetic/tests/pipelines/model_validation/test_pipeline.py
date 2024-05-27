import pytest
import pandas as pd
from unittest import mock
from keras.models import Model
from src.prothetic.pipelines.model_validation.nodes import evaluate_model
from src.prothetic.pipelines.model_validation.pipeline import create_pipeline

@pytest.fixture
def sample_test_data():
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50]
    }), pd.DataFrame({
        "target": [100, 200, 300, 400, 500]
    })

def test_evaluate_model(mocker, sample_test_data):
    X_test, y_test = sample_test_data
    mock_model = mocker.Mock(spec=Model)
    mock_model.evaluate.return_value = [0.1, 0.9]  # Example evaluation results

    mock_log_metrics = mocker.patch("src.prothetic.pipelines.model_validation.nodes.log_metrics")

    evaluation_metrics = evaluate_model(mock_model, X_test, y_test)

    mock_model.evaluate.assert_called_once_with(X_test, y_test, verbose=1)
    mock_log_metrics.assert_called_once_with({
        "loss": 0.1,
        "accuracy": 0.9
    })

    assert evaluation_metrics["loss"] == 0.1
    assert evaluation_metrics["accuracy"] == 0.9

def test_create_pipeline():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) == 1  # Check the number of nodes in the pipeline

    node_names = [node.name for node in pipeline.nodes]
    assert "evaluate_model_node" in node_names
