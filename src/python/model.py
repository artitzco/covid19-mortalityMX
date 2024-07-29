from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, ELU
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


def model_v1(input_dim,
             neurons,
             deep,
             optimizer,
             learning_rate,
             momentum,
             activation,
             activation_alpha,
             initializer,
             batch_size,
             epochs,
             batch_normalization,
             regularization,
             regularization_factor,
             dropout,
             dropout_rate):
    """
    Recordar condición:
        deep <= 1 + math.log(neurons / 2.0) / math.log(2)
    """

    # Optimizador

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'Nadam':
        optimizer = Nadam(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)

    # Inicializador

    if initializer == 'uniform':
        kernel_initializer = 'he_uniform'
    elif initializer == 'normal':
        kernel_initializer = 'he_normal'

    # Activación
    if activation == 'ELU':
        activation = ELU(activation_alpha)
    elif activation == 'LeakyReLU':
        activation = LeakyReLU(activation_alpha)

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

    return {
        'model': model,
        'fit': lambda train_set, val_set=None, verbose=1: model.fit(*train_set,
                                                                    validation_data=val_set,
                                                                    batch_size=batch_size,
                                                                    epochs=epochs,
                                                                    verbose=verbose)
    }


def model_v1_hparams(input_dim,
                     neurons,
                     deep,
                     optimizer,
                     optimizer_param,
                     activation,
                     activation_alpha,
                     initializer,
                     batch_size,
                     epochs,
                     batch_normalization,
                     regularization,
                     regularization_factor,
                     dropout,
                     dropout_rate):
    def select(value, cond, dtype=float):
        return dtype(value if cond else 0)

    return dict(input_dim=int(input_dim),
                neurons=int(neurons),
                deep=int(deep),
                optimizer=optimizer,
                learning_rate=(select(optimizer_param[0], optimizer == 'Adam') +
                               select(optimizer_param[1], optimizer == 'Nadam') +
                               select(optimizer_param[2][0], optimizer == 'RMSprop') +
                               select(optimizer_param[3][0], optimizer == 'SGD')),
                momentum=(select(optimizer_param[2][1], optimizer == 'RMSprop') +
                          select(optimizer_param[3][1], optimizer == 'SGD')),
                activation=activation,
                activation_alpha=(select(activation_alpha[0], activation == 'ELU') +
                                  select(activation_alpha[1], activation == 'LeakyReLU')),
                initializer=initializer,
                batch_size=int(batch_size),
                epochs=int(epochs),
                batch_normalization=batch_normalization,
                regularization=regularization,

                regularization_factor=(select(regularization_factor[0], regularization == 'l1') +
                                       select(regularization_factor[1], regularization == 'l2')),
                dropout=dropout,
                dropout_rate=select(dropout_rate, dropout))
