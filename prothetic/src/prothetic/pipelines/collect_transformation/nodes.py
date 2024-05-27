"""
This is a boilerplate pipeline 'collect_transformation'
generated using Kedro 0.19.5
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def transform_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the data by ensuring all values are floats and removing any rows with null values.
    """
    data_transformed = input_data.apply(pd.to_numeric, errors='coerce')

    data_transformed.dropna(inplace=True)
    data_transformed = data_transformed.drop(columns=['before_exam_125_Hz', 'before_exam_250_Hz', 'before_exam_500_Hz', 'before_exam_8000_Hz', 'after_exam_125_Hz', 'after_exam_250_Hz', 'after_exam_500_Hz', 'after_exam_8000_Hz'])

    return data_transformed


def split_dataset(input_data: pd.DataFrame):

    X = input_data.filter(regex='^before_')
    y = input_data.filter(regex='^after_')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
