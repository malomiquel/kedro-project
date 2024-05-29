from keras import layers, regularizers, optimizers, metrics, Model, callbacks
import pandas as pd
import mlflow

# Active l'enregistrement automatique avec MLflow
mlflow.autolog()

def create_model(input_shape, units=128, activation='relu', l2_value=0.01, dropout_rate=0.5, learning_rate=1e-3):
    """
    Crée un modèle de réseau de neurones avec les paramètres spécifiés.
    
    Args:
    - input_shape (tuple): La forme des données d'entrée.
    - units (int): Le nombre d'unités dans la couche dense.
    - activation (str): La fonction d'activation à utiliser.
    - l2_value (float): Le coefficient de régularisation L2.
    - dropout_rate (float): Le taux de dropout à utiliser.
    - learning_rate (float): Le taux d'apprentissage pour l'optimiseur.

    Returns:
    - model (Model): Le modèle de réseau de neurones compilé.
    """
    print(f"Creating model with input shape: {input_shape}")

    # Vérifie que la forme d'entrée a 3 dimensions
    if len(input_shape) != 3:
        raise ValueError(f"Expected input shape to have 3 dimensions, got {len(input_shape)} dimensions.")

    # Définition de la couche d'entrée
    inputs = layers.Input(shape=(input_shape[1], input_shape[2]))

    # Ajout des couches de convolution et de pooling
    x = layers.Conv1D(filters=64, kernel_size=3, activation=activation, padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation=activation, padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    # Aplatir les résultats et ajouter une couche dense avec régularisation L2
    x = layers.Flatten()(x)
    x = layers.Dense(units, activation=activation, kernel_regularizer=regularizers.L2(l2_value))(x)

    # Ajout de la couche de dropout si spécifié
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    # Définition de la couche de sortie avec activation linéaire
    outputs = layers.Dense(input_shape[1], activation='linear')(x)

    # Création du modèle
    model = Model(inputs=inputs, outputs=outputs)

    # Compilation du modèle avec l'optimiseur Adam et les métriques de performance
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss="mean_squared_error",
                  metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()])

    return model

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    """
    Entraîne le modèle sur les données d'entraînement et de validation.
    
    Args:
    - X_train (pd.DataFrame): Les features d'entraînement.
    - y_train (pd.DataFrame): Les labels d'entraînement.
    - X_test (pd.DataFrame): Les features de test.
    - y_test (pd.DataFrame): Les labels de test.

    Returns:
    - model (Model): Le modèle de réseau de neurones entraîné.
    """
    # Reshape des données pour qu'elles soient compatibles avec le modèle
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_train_reshaped = y_train.values.reshape((y_train.shape[0], y_train.shape[1], 1))
    y_test_reshaped = y_test.values.reshape((y_test.shape[0], y_test.shape[1], 1))

    # Création du modèle avec les données reshaped
    model = create_model(X_train_reshaped.shape, dropout_rate=0.5)

    # Définition du callback pour l'arrêt précoce
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Entraînement du modèle avec les données d'entraînement et de validation
    model.fit(X_train_reshaped, y_train_reshaped, epochs=10, validation_data=(X_test_reshaped, y_test_reshaped), callbacks=[early_stopping])

    return model
