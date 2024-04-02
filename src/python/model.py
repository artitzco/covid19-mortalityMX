import numpy as np
from tensorflow.keras.activations import relu, elu
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, SGD
from tensorflow.keras import regularizers


def classification_model(input_dim,
                         neurons,
                         deep,
                         optimizer,
                         learning_rate,
                         activation,
                         initializer,
                         batch_size,
                         epochs,
                         batch_normalization,
                         regularization,
                         regularization_factor,
                         dropout,
                         dropout_rate):

    deep = int(max(min(1 + np.log(neurons / 2.0) / np.log(2),  deep), 1))

    # Optimizador

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'Nadam':
        optimizer = Nadam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)

    # Inicializador

    if initializer == 'uniform':
        kernel_initializer = 'he_uniform'
    elif initializer == 'normal':
        kernel_initializer = 'he_normal'

    # Activación
    if activation == 'relu':
        activation = relu
    elif activation == 'elu':
        activation = elu
    elif activation == 'LeakyReLU':
        activation = LeakyReLU()

    # Regularización
    if regularization == 'l1':
        kernel_regularizer = regularizers.l1(regularization_factor)
    elif regularization == 'l2':
        kernel_regularizer = regularizers.l2(regularization_factor)
    elif regularization == 'None':
        kernel_regularizer = None

    # Modelo
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(neurons,
                    activation=activation,
                    kernel_regularizer=kernel_regularizer,
                    kernel_initializer=kernel_initializer))

    model.add(BatchNormalization()) if batch_normalization else None
    model.add(Dropout(dropout_rate)) if dropout else None

    for x in range(deep - 1):
        model.add(Dense(neurons//(2)**(1+x),
                        activation=activation,
                        kernel_regularizer=kernel_regularizer,
                        kernel_initializer=kernel_initializer))
        model.add(BatchNormalization()) if batch_normalization else None
        model.add(Dropout(dropout_rate)) if dropout else None

    model.add(Dense(1, activation='sigmoid',
                    kernel_initializer=('glorot_uniform' if initializer == 'uniform'
                                        else ('glorot_normal' if initializer == 'normal' else ''))))

    # Compilar el modelo
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mae'])

    return model, lambda train_set, val_set=None, verbose=1: model.fit(*train_set,
                                                                       validation_data=val_set,
                                                                       batch_size=batch_size,
                                                                       epochs=epochs,
                                                                       verbose=verbose)


def validate_classification_model_hparams(hparam):

    hparam['input_dim'] = int(hparam['input_dim'])
    hparam['neurons'] = int(hparam['neurons'])
    hparam['deep'] = int(
        max(min(1 + np.log(hparam['neurons'] / 2.0) / np.log(2),  hparam['deep']), 1))
    hparam['learning_rate'] = float(hparam['learning_rate'])
    hparam['batch_size'] = int(hparam['batch_size'])
    hparam['epochs'] = int(hparam['epochs'])

    hparam['regularization_factor'] = float(hparam['regularization_factor']
                                            if hparam['regularization'] != 'None' else 0)

    hparam['dropout_rate'] = float(hparam['dropout_rate']
                                   if hparam['dropout'] else 0)
