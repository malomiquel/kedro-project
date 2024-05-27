import pytest
import pandas as pd
from unittest import mock
from keras.models import Model
from src.prothetic.pipelines.model_training.nodes import create_model, train_model
from src.prothetic.pipelines.model_training.pipeline import create_pipeline

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50]
    }), pd.DataFrame({
        "target": [100, 200, 300, 400, 500]
    })

def test_create_model():
    input_shape = (10, 5, 1)
    model = create_model(input_shape)
    assert isinstance(model, Model)
    assert model.input_shape == (None, 5, 1)
    assert model.output_shape == (None, 5)

def test_train_model(mocker, sample_data):
    X, y = sample_data
    mock_model = mocker.Mock()
    mock_create_model = mocker.patch("src.prothetic.pipelines.model_training.nodes.create_model", return_value=mock_model)
    mock_early_stopping = mocker.patch("src.prothetic.pipelines.model_training.nodes.callbacks.EarlyStopping", return_value=mocker.Mock())

    x_train, x_test = X.iloc[:3], X.iloc[3:]
    y_train, y_test = y.iloc[:3], y.iloc[3:]

    model = train_model(x_train, y_train, x_test, y_test)

    # Verify if the model was created and trained
    mock_create_model.assert_called_once()
    mock_model.fit.assert_called_once()
    assert model == mock_model

def test_create_pipeline():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) == 1  # Check the number of nodes in the pipeline

    node_names = [node.name for node in pipeline.nodes]
    assert "node_train_model" in node_names

def test_create_model_invalid_shape():
    invalid_input_shape = (10, 5)  # Only 2 dimensions instead of 3
    with pytest.raises(ValueError, match="Expected input shape to have 3 dimensions"):
        create_model(invalid_input_shape)
