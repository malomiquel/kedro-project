from keras import layers, regularizers, optimizers, metrics, Model, callbacks
import pandas as pd
import mlflow

mlflow.autolog()

def create_model(input_shape, units=128, activation='relu', l2_value=0.01, dropout_rate=0.5, learning_rate=1e-3):
    print(f"Creating model with input shape: {input_shape}")

    # Ensure input_shape has three dimensions
    if len(input_shape) != 3:
        raise ValueError(f"Expected input shape to have 3 dimensions, got {len(input_shape)} dimensions.")

    inputs = layers.Input(shape=(input_shape[1], input_shape[2]))

    x = layers.Conv1D(filters=64, kernel_size=3, activation=activation, padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation=activation, padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units, activation=activation, kernel_regularizer=regularizers.L2(l2_value))(x)

    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(input_shape[1], activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss="mean_squared_error",
                  metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()])

    return model

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    # Reshape for Conv1D input
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_train_reshaped = y_train.values.reshape((y_train.shape[0], y_train.shape[1], 1))
    y_test_reshaped = y_test.values.reshape((y_test.shape[0], y_test.shape[1], 1))

    # Debug prints to check shapes
    print(f"X_train reshaped shape: {X_train_reshaped.shape}")
    print(f"X_test reshaped shape: {X_test_reshaped.shape}")
    print(f"y_train reshaped shape: {y_train_reshaped.shape}")
    print(f"y_test reshaped shape: {y_test_reshaped.shape}")

    model = create_model(X_train_reshaped.shape, dropout_rate=0.5)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train_reshaped, y_train_reshaped, epochs=100, validation_data=(X_test_reshaped, y_test_reshaped), callbacks=[early_stopping])

    return model  # Return only the model