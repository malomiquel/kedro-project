"""
This is a boilerplate pipeline 'collect_transformation'
generated using Kedro 0.19.5
"""

import pandas as pd

def transform_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the data by ensuring all values are floats and removing any rows with null values.
    """
    data_transformed = input_data.apply(pd.to_numeric, errors='coerce')

    data_transformed.dropna(inplace=True)

    return data_transformed
