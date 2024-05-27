import pytest
import pandas as pd
from unittest import mock
from sklearn.linear_model import LinearRegression
from src.prothetic.pipelines.model_deployment.nodes import choose_best_model
from src.prothetic.pipelines.model_deployment.pipeline import create_pipeline

@pytest.fixture
def sample_test_data():
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50]
    }), pd.DataFrame({
        "target": [100, 200, 300, 400, 500]
    })

def test_choose_best_model_new_model(mocker, sample_test_data):
    X_test, y_test = sample_test_data
    new_model = mocker.Mock(spec=LinearRegression)
    new_model.predict.return_value = [110, 210, 310, 410, 510]

    old_model_path = "old_model.pkl"
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("pickle.load", return_value=new_model)  # Mock the same new model as old model for simplicity

    chosen_model = choose_best_model(X_test, y_test, new_model, old_model_path)

    assert chosen_model == new_model

def test_choose_best_model_old_model(mocker, sample_test_data):
    X_test, y_test = sample_test_data
    new_model = mocker.Mock(spec=LinearRegression)
    new_model.predict.return_value = [110, 210, 310, 410, 510]
    
    old_model = mocker.Mock(spec=LinearRegression)
    old_model.predict.return_value = [105, 205, 305, 405, 505]

    old_model_path = "old_model.pkl"
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("pickle.load", return_value=old_model)

    chosen_model = choose_best_model(X_test, y_test, new_model, old_model_path)

    assert chosen_model == old_model

def test_choose_best_model_no_old_model(mocker, sample_test_data):
    X_test, y_test = sample_test_data
    new_model = mocker.Mock(spec=LinearRegression)
    new_model.predict.return_value = [110, 210, 310, 410, 510]

    old_model_path = "non_existent_model.pkl"
    mocker.patch("builtins.open", side_effect=FileNotFoundError)

    chosen_model = choose_best_model(X_test, y_test, new_model, old_model_path)

    assert chosen_model == new_model

def test_create_pipeline():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) == 1  # Check the number of nodes in the pipeline

    node_names = [node.name for node in pipeline.nodes]
    assert "node_choose_best_model" in node_names
