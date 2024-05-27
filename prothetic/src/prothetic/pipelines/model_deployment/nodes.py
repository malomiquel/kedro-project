"""
This is a boilerplate pipeline 'model_deployment'
generated using Kedro 0.19.5
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

def choose_best_model(X_test, y_test, new_model, old_model_path):

    try:
        with open(old_model_path, 'rb') as f:
            old_model = pickle.load(f)
    except FileNotFoundError:
        return new_model

    y_pred = new_model.predict(X_test)
    new_mae = mean_absolute_error(y_test, y_pred)
    new_mse = mean_squared_error(y_test, y_pred)

    y_pred_old = old_model.predict(X_test)
    old_mae = mean_absolute_error(y_test, y_pred_old)
    old_mse = mean_squared_error(y_test, y_pred_old)

    if new_mse > old_mse or new_mae > old_mae:
        return old_model
    else:
        return new_model