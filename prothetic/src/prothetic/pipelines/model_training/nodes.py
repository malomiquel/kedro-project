from keras import layers, regularizers, optimizers, metrics, Model

def create_model(input_data, units=128, activation='relu', l2_value=0.01, dropout_rate=None, learning_rate=1e-3):
    
    input_shape = input_data.shape

    # Définition de la couche d'entrée
    inputs = layers.Input(shape=input_shape) # format (dim,1)

    # Définition des couches de convolution
    x = layers.Conv1D(filters=32, kernel_size=3, activation=activation)(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.ZeroPadding1D(padding=1)(x)  # Ajouter une couche de padding
    x = layers.Conv1D(filters=64, kernel_size=3, activation=activation)(x)
    x = layers.ZeroPadding1D(padding=1)(x)  # Ajouter une couche de padding
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Aplatir les données
    x = layers.Flatten()(x)

    # Définition des couches entièrement connectées
    x = layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_value))(x)
    
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(input_shape[0], activation='softmax')(x)

    # Création du modèle
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss="mse", metrics=[metrics.CategoricalAccuracy()])
    
    return model