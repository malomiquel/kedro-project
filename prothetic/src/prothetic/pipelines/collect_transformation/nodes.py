import pandas as pd
from sklearn.model_selection import train_test_split

def transform_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme les données en s'assurant que toutes les valeurs sont des floats et en supprimant les lignes contenant des valeurs nulles.
    """
    # Colonnes nécessaires pour l'entraînement du modèle
    columns = [
        'before_exam_125_Hz', 'before_exam_250_Hz', 'before_exam_500_Hz', 'before_exam_1000_Hz',
        'before_exam_2000_Hz', 'before_exam_4000_Hz', 'before_exam_8000_Hz', 'after_exam_125_Hz',
        'after_exam_250_Hz', 'after_exam_500_Hz', 'after_exam_1000_Hz', 'after_exam_2000_Hz',
        'after_exam_4000_Hz', 'after_exam_8000_Hz'
    ]

    # Vérifie que les données d'entrée contiennent toutes les colonnes nécessaires
    missing_columns = [col for col in columns if col not in input_data.columns]
    if missing_columns:
        raise ValueError(f"Les colonnes suivantes sont manquantes dans les données d'entrée : {', '.join(missing_columns)}")

    # Convertit toutes les valeurs du DataFrame en floats, les erreurs sont converties en NaN
    data_transformed = input_data.apply(pd.to_numeric, errors='coerce')
    
    # Supprime les lignes contenant des valeurs nulles
    data_transformed.dropna(inplace=True)

    # Colonnes à supprimer du DataFrame
    columns_to_drop = [
        'before_exam_125_Hz', 'before_exam_250_Hz', 'before_exam_500_Hz',
        'before_exam_8000_Hz', 'after_exam_125_Hz', 'after_exam_250_Hz',
        'after_exam_500_Hz', 'after_exam_8000_Hz'
    ]
    
    # Supprime les colonnes spécifiées du DataFrame
    data_transformed.drop(columns=columns_to_drop, inplace=True)

    return data_transformed

def split_dataset(input_data: pd.DataFrame):
    # Sépare les données en features (X) et labels (y) en filtrant les colonnes correspondant aux regex 'before_' et 'after_'
    X = input_data.filter(regex='^before_')
    y = input_data.filter(regex='^after_')

    # Divise les données en ensembles d'entraînement et de test avec un ratio de test de 20% et une graine aléatoire pour la reproductibilité
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertit les résultats des ensembles d'entraînement et de test en DataFrames avec les mêmes colonnes que les DataFrames originaux
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.DataFrame(y_train, columns=y.columns)
    y_test = pd.DataFrame(y_test, columns=y.columns)

    return X_train, X_test, y_train, y_test