"""
This is a boilerplate pipeline 'model_prediction'
generated using Kedro 0.19.5
"""

from keras import Model
import pandas as pd


def transform_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the data by ensuring all values are floats and removing any rows with null values.
    Keep only the necessary columns for prediction.
    """

    prediction_columns = [
        'before_exam_1000_Hz', 'before_exam_2000_Hz', 'before_exam_4000_Hz'
    ]

    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")

    missing_columns = [
        col for col in prediction_columns if col not in input_data.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the input data: {
                         ', '.join(missing_columns)}")

    data_transformed = input_data.apply(pd.to_numeric, errors='coerce')
    data_transformed.dropna(inplace=True)

    data_transformed = data_transformed[prediction_columns]

    return data_transformed


def predict_model(input_data: pd.DataFrame, model: Model) -> pd.DataFrame:
    """
    Predict the output of the model using the input data.
    """

    data_predicted = model.predict(input_data)

    df = pd.DataFrame(data_predicted, columns=[
                      'after_exam_1000_Hz', 'after_exam_2000_Hz', 'after_exam_4000_Hz'])

    df = df.round().astype(int)

    return df
