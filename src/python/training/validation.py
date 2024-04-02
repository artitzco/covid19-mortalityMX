import os.path as path
import pandas as pd
import numpy as np
import itertools
import hashlib
import pickle
import json
import time


def _hashing(dictionary):
    dictionary = {k: str(v) for k, v in dictionary.items() if v is not None}
    return hashlib.sha256(json.dumps(dictionary, sort_keys=True).encode()).hexdigest()


class Fold:
    def __init__(self, origin, hparam):
        self.origin = origin
        self.hparam = hparam
        self.history = []
        self.time = []
        self.summary = []

    def add(self, history, time):
        self.history.append(history)
        self.time.append(time)
        self.summary.append(f"time: {int(time // 3600)}h {int((time % 3600) // 60)}m {int(time % 60)}s   " +
                            '   '.join([f'{k}: {v:.6f}' for k, v in history.iloc[-1].items()]))

    def total(self, train_size, val_size):
        train_prop = np.array(train_size)/sum(train_size)
        val_prop = np.array(val_size)/sum(val_size)

        history = 0
        for i, hist in enumerate(self.history):
            hist = hist.copy()
            for name, value in hist.items():
                prop = (val_prop[i] if name.startswith(
                    'val_') else train_prop[i])
                hist[name] = value * prop
            history += hist
        self.history.append(history)
        time = sum(self.time)
        self.time.append(time)
        self.summary.append(f"time: {int(time // 3600)}h {int((time % 3600) // 60)}m {int(time % 60)}s   " +
                            '   '.join([f'{k}: {v:.6f}' for k, v in history.iloc[-1].items()]))
        for hist in self.history:
            hist.drop(['loss', 'val_loss'], axis=1, inplace=True)

    @property
    def dictionary(self):
        return dict(origin=self.origin,
                    hparam=self.hparam,
                    history=self.history,
                    time=self.time,
                    summary=self.summary)


class Foldify:
    def __init__(self,
                 size,
                 nfolds=5,
                 val_prop=0.2,
                 ftype='random',
                 sorted=True,
                 seed=None,
                 weight=None,
                 label=None,
                 datasets=None):
        size = int(size)
        nfolds = int(nfolds)
        val_prop = float(val_prop)
        if ftype == 'uniform':
            nval = int(round(size * val_prop))
            seed, weight, label = None, None, None
        elif ftype == 'random':
            self.random = np.random.RandomState()
            if label is not None:
                sub_label = label[np.arange(size)]
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
        self.sorted = sorted
        self.seed = seed
        self.weight = weight
        self.label = label
        self.datasets = datasets
        self.train_size = train_size
        self.val_size = val_size

    @property
    def info(self):
        return dict(size=self.size,
                    nfolds=self.nfolds,
                    val_prop=self.val_prop,
                    ftype=self.ftype,
                    sorted=self.sorted,
                    seed=self.seed,
                    weight=self.weight is not None,
                    label=self.label is not None,
                    train_size=self.train_size,
                    val_size=self.val_size)

    @property
    def hash(self):
        return _hashing(self.info)

    def _uniform_val_index(self, index):
        return np.array([], dtype=int)

    def _stand_weight(self, index):
        if self.weight is None:
            return None
        weight = self.weight[index]
        return weight / sum(weight)

    def _random_val_index(self, index):
        if self.label is None:
            weight = self._stand_weight(index)
            nval = int(round(self.size * self.val_prop))
            val_index = self.random.choice(
                index, size=nval, replace=False, p=weight)
        else:
            label = self.label[index]
            val_index = np.array([], dtype=int)
            for lab in np.unique(label):
                sub_index = index[label == lab]
                weight = self._stand_weight(sub_index)
                nval = int(round(len(sub_index) * self.val_prop))
                lab_val_index = self.random.choice(
                    sub_index, size=nval, replace=False, p=weight)
                val_index = np.concatenate((val_index, lab_val_index))
        val_index.sort()
        return val_index

    def get(self, i):
        index = np.arange(self.size)
        if self.ftype == 'uniform':
            val_index = self._uniform_val_index(index)

        elif self.ftype == 'random':
            self.random.seed(None if self.seed is None else self.seed + i)
            val_index = self._random_val_index(index)

        train = np.ones(self.size, dtype=bool)
        train[val_index] = False
        train_index = index[train]
        if not self.sorted:
            self.random.shuffle(train_index)
            self.random.shuffle(val_index)
        if self.datasets is None:
            return dict(train_index=train_index, val_index=val_index)
        return dict(
            train_set=(self.datasets[0][train_index],
                       self.datasets[1][train_index]),
            val_set=(self.datasets[0][val_index],
                     self.datasets[1][val_index]))


class Memory:

    def __init__(self, input):
        if isinstance(input, Foldify):
            self.hash = input.hash
            self.finfo = input.info
            self.data = dict()
        else:
            self.hash = input['hash']
            self.finfo = input['finfo']
            self.data = input['data']

    @property
    def dictionary(self):
        return dict(hash=self.hash,
                    finfo=self.finfo,
                    data=self.data)

    @property
    def hparam(self):
        return pd.DataFrame([value['hparam'] for value in self.data.values()])

    @property
    def metric(self):
        row = []
        for value in self.data.values():
            dic = dict(value['history'][-1].iloc[-1])
            dic['time'] = value['time'][-1]
            row.append(dic)
        return pd.DataFrame(row)

    def groups(self, hparam, metric):
        df = self.hparam
        data = pd.DataFrame({hparam: df.pop(hparam),
                             metric: self.metric[metric]})

        hash = []
        for i in range(len(df)):
            hash.append(_hashing(dict(df.iloc[i])))
        df.index = hash
        df = df[~df.index.duplicated(keep='first')]
        data.index = hash

        subdata = []
        for index in df.index:
            subdata.append(data[data.index == index].sort_values(hparam))
            subdata[-1].reset_index(drop=True, inplace=True)

        df['subdata'] = subdata
        df.reset_index(drop=True, inplace=True)
        return df


def eval_model(model, hparam, foldify, memory=None, infile=None, outfile=None, origin='unknown'):
    print('Hyperparameters:\n\t',
          '   '.join([f'{k}: {v}' for k, v in hparam.items()]))

    if infile is not None and path.exists(infile):
        with open(infile, "rb") as file:
            memory = Memory(pickle.load(file))
    else:
        memory = Memory(foldify) if memory is None else memory
    if memory.hash != foldify.hash:
        raise ValueError('memory hash must be equals to foldify hash.')
    data = memory.data
    hash = _hashing(hparam)
    if hash in data:
        summary_list = data[hash]['summary']
        for i, summary in enumerate(summary_list[:-1]):
            print(f'Fold {i+1}/{foldify.nfolds} ...')
            print('\t', summary)
        print('Total:\n\t', summary_list[-1])
        return data[hash].copy()

    fold = Fold(origin, hparam)
    for i in range(foldify.nfolds):
        print(f'Fold {i+1}/{foldify.nfolds} ...')
        datasets = foldify.get(i)
        _, fit = model(**hparam)
        start = time.time()
        history = pd.DataFrame(fit(train_set=datasets['train_set'],
                                   val_set=datasets['val_set'],
                                   verbose=0).history)
        fold.add(history, time.time() - start)
        print('\t', fold.summary[-1])
    fold.total(foldify.train_size, foldify.val_size)
    print('Total:\n\t', fold.summary[-1])
    data[hash] = fold.dictionary

    if outfile is not None:
        with open(outfile, "wb") as file:
            pickle.dump(memory.dictionary, file)

    return data[hash].copy()


def make_grid(hparams, validation):
    grid = dict()
    for hparams in [{k: v for k, v in zip(hparams.keys(), lst)} for lst in
                    list(itertools.product(*[v if isinstance(v, list) else [v] for v in hparams.values()]))]:
        validation(hparams)
        hash = _hashing(hparams)

        if hash not in grid:
            grid[hash] = hparams
    return list(grid.values())
