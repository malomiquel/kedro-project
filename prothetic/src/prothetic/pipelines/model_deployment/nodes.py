"""
This is a boilerplate pipeline 'model_deployment'
generated using Kedro 0.19.5
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

def choose_best_model(X_test, y_test, new_model, old_model_path):
    """
    Choisit le meilleur modèle entre un nouveau modèle et un ancien modèle sauvegardé.
    
    Args:
    - X_test (pd.DataFrame): Les données de test pour les features.
    - y_test (pd.DataFrame): Les données de test pour les labels.
    - new_model: Le nouveau modèle entraîné.
    - old_model_path (str): Le chemin vers l'ancien modèle sauvegardé.

    Returns:
    - Le meilleur modèle basé sur les erreurs moyennes absolues et les erreurs quadratiques moyennes.
    """
    
    try:
        # Essaie de charger l'ancien modèle à partir du chemin donné
        with open(old_model_path, 'rb') as f:
            old_model = pickle.load(f)
    except FileNotFoundError:
        # Si le fichier de l'ancien modèle n'existe pas, retourne le nouveau modèle
        return new_model

    # Prédit les valeurs cibles avec le nouveau modèle et calcule les métriques
    y_pred = new_model.predict(X_test)
    new_mae = mean_absolute_error(y_test, y_pred)
    new_mse = mean_squared_error(y_test, y_pred)

    # Prédit les valeurs cibles avec l'ancien modèle et calcule les mêmes métriques
    y_pred_old = old_model.predict(X_test)
    old_mae = mean_absolute_error(y_test, y_pred_old)
    old_mse = mean_squared_error(y_test, y_pred_old)

    # Compare les métriques des deux modèles et retourne le modèle ayant les meilleures performances
    if new_mse > old_mse or new_mae > old_mae:
        return old_model
    else:
        return new_model