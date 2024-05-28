"""
This is a boilerplate pipeline 'model_prediction'
generated using Kedro 0.19.5
"""

from keras import Model
import pandas as pd

def transform_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the data by ensuring all values are floats and removing any rows with null values.
    """
    data_transformed = input_data.apply(pd.to_numeric, errors='coerce')
    data_transformed.dropna(inplace=True)

    columns_to_drop = [
        'before_exam_125_Hz', 'before_exam_250_Hz', 'before_exam_500_Hz',
        'before_exam_8000_Hz', 'after_exam_125_Hz', 'after_exam_250_Hz',
        'after_exam_500_Hz', 'after_exam_8000_Hz'
    ]
    data_transformed.drop(columns=columns_to_drop, inplace=True)

    return data_transformed

def predict_model(input_data: pd.DataFrame, model: Model) -> pd.DataFrame:
    """
    Predict the output of the model using the input data.
    """
    data_predicted = model.predict(input_data)

    return data_predicted