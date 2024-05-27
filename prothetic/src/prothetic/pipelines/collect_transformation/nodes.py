import pandas as pd
from sklearn.model_selection import train_test_split

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

def split_dataset(input_data: pd.DataFrame):
    X = input_data.filter(regex='^before_')
    y = input_data.filter(regex='^after_')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert back to DataFrame to save
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.DataFrame(y_train, columns=y.columns)
    y_test = pd.DataFrame(y_test, columns=y.columns)

    # Debug prints to check shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test
