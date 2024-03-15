from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers


from src.python.util import save_object
from src.python.util import load_object
from src.python.util import dict_hash
from os import path
import pandas as pd
import numpy as np
import itertools
import time


class Fold:
    def __init__(self,
                 size,
                 nfolds=5,
                 val_prop=0.2,
                 ftype='random',
                 seed=None,
                 weight=None,
                 label=None,
                 datasets=None):
        size = int(size)
        nfolds = int(nfolds)
        val_prop = float(val_prop)

        ############
        if ftype == 'uniform':
            nval = int(round(size * val_prop))
            train_size = [size - nval] * nfolds
            val_size = [nval] * nfolds
            seed, weight, label = None, None, None
        elif ftype == 'random':
            index = np.arange(size)
            if label is not None:
                sub_label = label[index]
                nval = 0
                for lab in np.unique(sub_label):
                    nval += int(round(sum(sub_label == lab) * val_prop))
            else:
                nval = int(round(size * val_prop))
            train_size = [size - nval] * nfolds
            val_size = [nval] * nfolds

        self.size = size
        self.nfolds = nfolds
        self.val_prop = val_prop
        self.ftype = ftype
        self.seed = seed
        self.weight = weight
        self.label = label
        self.datasets = datasets
        self.train_size = train_size
        self.val_size = val_size

    @property
    def info(self):
        return dict(
            size=self.size,
            nfolds=self.nfolds,
            val_prop=self.val_prop,
            ftype=self.ftype,
            seed=self.seed,
            weight=self.weight is not None,
            label=self.label is not None,
            train_size=self.train_size,
            val_size=self.val_size)

    def _norm_weight(self, index):
        if self.weight is None:
            return None
        weight = self.weight[index]
        weight = weight / sum(weight)
        return weight

    def get(self, i, sorted=True):
        index = np.arange(self.size)
        if self.ftype == 'uniform':
            val_index = np.array([], dtype=int)
        elif self.ftype == 'random':
            np.random.seed(None if self.seed is None else self.seed + i)
            if self.label is None:
                weight = self._norm_weight(index)
                nval = int(round(self.size * self.val_prop))
                val_index = np.random.choice(
                    index, size=nval, replace=False, p=weight)
            else:
                label = self.label[index]
                val_index = np.array([], dtype=int)
                for lab in np.unique(label):
                    sub_index = index[label == lab]
                    weight = self._norm_weight(sub_index)
                    nval = int(round(len(sub_index) * self.val_prop))
                    val_index = np.concatenate((val_index, np.random.choice(
                        sub_index, size=nval, replace=False, p=weight)))
            val_index.sort()

        train = np.ones(self.size, dtype=bool)
        train[val_index] = False
        train_index = index[train]
        if not sorted:
            np.random.shuffle(train_index)
            np.random.shuffle(val_index)
        if self.datasets is None:
            return dict(train_index=train_index, val_index=val_index)
        return dict(
            train_set=(self.datasets[0][train_index],
                       self.datasets[1][train_index]),
            val_set=(self.datasets[0][val_index],
                     self.datasets[1][val_index]))


def new_memory(fold):
    memory = dict(fold=fold.info, data=dict())
    memory['fold']['hash'] = dict_hash(memory['fold'])
    return memory


def eval_model(model, hparam, fold, file=None, memory=None):
    validate_classification_model_hparams(hparam)
    print('Hyperparameters:\n\t' +
          '   '.join([f'{k}: {v}' for k, v in hparam.items()]))

    memory = new_memory(fold) if memory is None else memory

    if memory['fold']['hash'] != dict_hash(fold.info):
        raise ValueError('Incompatible fold!!!')

    phash = dict_hash(hparam)
    if phash in memory['data']:
        info = memory['data'][phash]
        for i in range(fold.nfolds):
            print(f'Fold {i+1}/{fold.nfolds} ...')
            print(info['summary'][i])
        print('Total')
        print(info['summary'][-1], '\n')
        None if file is None else save_object(memory, file)
        return info
    else:
        memory['data'][phash] = dict(hparam=hparam)
        info = memory['data'][phash]

    train_prop = np.array(fold.train_size)/sum(fold.train_size)
    val_prop = np.array(fold.val_size)/sum(fold.val_size)
    history, total, seconds, summary = [], 0, [], []
    for i in range(fold.nfolds):
        print(f'Fold {i+1}/{fold.nfolds} ...')
        datasets = fold.get(i)
        _, fit = model(**hparam)
        start = time.time()
        hist = pd.DataFrame(fit(train_set=datasets['train_set'],
                                val_set=datasets['val_set'],
                                verbose=0).history)
        seconds.append(time.time()-start)
        history.append(hist.copy())
        for name, value in hist.items():
            prop = val_prop if name.startswith('val_') else train_prop
            hist[name] = value * prop[i]
        total = total + hist
        # salida
        output = f"\ttime: {int(seconds[-1] // 3600)}h {int((seconds[-1] % 3600) // 60)}m {int(seconds[-1] % 60)}s"
        output += '   ' + \
            '   '.join(
                [f'{k}: {v:.6f}' for k, v in history[-1].iloc[-1].items()])
        summary.append(output)
        print(output)
    history.append(total)
    seconds.append(sum(seconds))
    print('Total')
    output = f"\ttime: {int(seconds[-1] // 3600)}h {int((seconds[-1] % 3600) // 60)}m {int(seconds[-1] % 60)}s"
    output += '   ' + \
        '   '.join(
            [f'{k}: {v:.6f}' for k, v in history[-1].iloc[-1].items()])
    summary.append(output)
    print(output, '\n')
    info['history'] = history
    info['seconds'] = seconds
    info['summary'] = summary
    None if file is None else save_object(memory, file)
    return info


def make_grid(hparams, validation):
    grid = dict()
    for hparams in [{k: v for k, v in zip(hparams.keys(), lst)} for lst in
                    list(itertools.product(*[v if isinstance(v, list) else [v] for v in hparams.values()]))]:
        validation(hparams)
        hash = dict_hash(hparams)
        if hash not in grid:
            grid[hash] = hparams
    return list(grid.values())


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

    # Regularización
    if regularization == 'l1':
        kernel_regularizer = regularizers.l1(regularization_factor)
    elif regularization == 'l2':
        kernel_regularizer = regularizers.l2(regularization_factor)
    elif regularization == 'None':
        kernel_regularizer = None

    # Modelo
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim,
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
