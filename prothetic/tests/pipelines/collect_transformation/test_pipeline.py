import pytest
import pandas as pd
from unittest import mock
from src.prothetic.pipelines.collect_transformation.nodes import transform_data, split_dataset
from src.prothetic.pipelines.collect_transformation.pipeline import create_pipeline

@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "target": [100, 200, 300, 400, 500],
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
def shaped_data():
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50]
    }), pd.DataFrame({
        "target": [100, 200, 300, 400, 500]
    })

def test_transform_data(sample_raw_data):
    transformed_data = transform_data(sample_raw_data)
    assert transformed_data is not None
    assert isinstance(transformed_data, pd.DataFrame)

def test_split_dataset(shaped_data):
    data = pd.concat([shaped_data[0], shaped_data[1]], axis=1)
    x_train, x_test, y_train, y_test = split_dataset(data)
    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_create_pipeline():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) == 2  # Check the number of nodes in the pipeline

    node_names = [node.name for node in pipeline.nodes]
    assert "node_merge_raw_daily_data" in node_names
    assert "node_split_transform_daily_data" in node_names
