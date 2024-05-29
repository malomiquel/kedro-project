"""
This is a boilerplate pipeline 'model_prediction'
generated using Kedro 0.19.5
"""

from keras import Model
import pandas as pd

def transform_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme les données en s'assurant que toutes les valeurs sont des floats et en supprimant les lignes contenant des valeurs nulles.
    Conserve uniquement les colonnes nécessaires pour la prédiction.
    """

    # Colonnes nécessaires pour la prédiction
    prediction_columns = [
        'before_exam_1000_Hz', 'before_exam_2000_Hz', 'before_exam_4000_Hz'
    ]

    # Vérifie que les données d'entrée sont un DataFrame pandas
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("Les données d'entrée doivent être un DataFrame pandas")

    # Vérifie que toutes les colonnes nécessaires sont présentes dans les données d'entrée
    missing_columns = [
        col for col in prediction_columns if col not in input_data.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the input data: {', '.join(missing_columns)}")

    # Convertit toutes les valeurs en floats et supprime les lignes contenant des valeurs nulles
    data_transformed = input_data.apply(pd.to_numeric, errors='coerce')
    data_transformed.dropna(inplace=True)

    # Conserve uniquement les colonnes nécessaires pour la prédiction
    data_transformed = data_transformed[prediction_columns]

    return data_transformed

def predict_model(input_data: pd.DataFrame, model: Model) -> pd.DataFrame:
    """
    Prédit la sortie du modèle en utilisant les données d'entrée.
    """

    # Utilise le modèle pour prédire les valeurs cibles à partir des données d'entrée
    data_predicted = model.predict(input_data)

    # Crée un DataFrame avec les prédictions, en nommant les colonnes correspondantes
    df = pd.DataFrame(data_predicted, columns=[
                      'after_exam_1000_Hz', 'after_exam_2000_Hz', 'after_exam_4000_Hz'])

    # Arrondit les valeurs prédites et les convertit en entiers
    df = df.round().astype(int)

    return df
