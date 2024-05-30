import pytest
import pandas as pd
from unittest import mock
from sklearn.linear_model import LinearRegression
from src.prothetic.pipelines.model_prediction.nodes import transform_data, predict_model
from src.prothetic.pipelines.model_prediction.pipeline import create_pipeline

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "before_exam_1000_Hz": [1, "A", 3, 4, 5],
        "before_exam_2000_Hz": [1, 2, 3, 4, 5],
        "before_exam_4000_Hz": [1, 2, 3, 4, 5],
        "after_exam_1000_Hz": [1, 2, 3, 4, 5],
        "after_exam_2000_Hz": [1, 2, 3, 5, ""],
        "after_exam_4000_Hz": [1, 2, 3, 4, 5],
        "before_exam_125_Hz": [1, 2, 3, 4, 5],
        "before_exam_250_Hz": [1, 2, 3, 4, 5],
        "before_exam_500_Hz": [1, 2, 3, 4, 5],
        "before_exam_8000_Hz": [1, 2, 3, 4, 5],
        "after_exam_125_Hz": [1, 2, 3, 4, 5],
        "after_exam_250_Hz": [1, 2, 3, 4, 5],
        "after_exam_500_Hz": [1, 2, 3, 4, 5],
        "after_exam_8000_Hz": [1, 2, 3, 4, 5]
    })

@pytest.fixture
def bad_sample():
    return pd.DataFrame({
        'invalid_column': [1.0, 2.0, 3.0]
    })

@pytest.fixture
def bad_sample_type():
    return "invalid type"

@pytest.fixture
def valid_data_predict():
    return pd.DataFrame({
        "before_exam_1000_Hz": [1, 2, 3, 4, 5],
        "before_exam_2000_Hz": [1, 2, 3, 4, 5],
        "before_exam_4000_Hz": [1, 2, 3, 4, 5]
    })

@pytest.fixture
def mocked_model():
    # Crée un objet MagicMock pour simuler le modèle
    model = mock.MagicMock()

    # Configure le comportement simulé du modèle pour renvoyer des prédictions factices
    model.predict.return_value = [
        [10.5, 20.5, 30.5],
        [11.5, 21.5, 31.5],
        [12.5, 22.5, 32.5]
    ]

    return model


def test_transform_data_prediction(sample_data):
    transformed_data = transform_data(sample_data)
    assert isinstance(transformed_data, pd.DataFrame)
    assert transformed_data.shape == (3, 3)

def test_transform_data_invalid(bad_sample):
    with pytest.raises(ValueError, match="The following required columns are missing from the input data: before_exam_1000_Hz, before_exam_2000_Hz, before_exam_4000_Hz"):
        transform_data(bad_sample)

def test_transform_data_type_invalid(bad_sample_type):
    with pytest.raises(TypeError, match="Les données d'entrée doivent être un DataFrame pandas"):
        transform_data(bad_sample_type)

def test_create_pipeline():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) == 2  # Check the number of nodes in the pipeline

    node_names = [node.name for node in pipeline.nodes]
    assert "node_transform_data_to_predict" in node_names
    assert "node_predict_model" in node_names

def test_predict_model(valid_data_predict, mocked_model):
    predicted_data = predict_model(valid_data_predict, mocked_model)
    assert isinstance(predicted_data, pd.DataFrame)
    assert predicted_data.shape == (3, 3)
    assert predicted_data.dtypes.apply(lambda x: pd.api.types.is_integer_dtype(x)).all() == True