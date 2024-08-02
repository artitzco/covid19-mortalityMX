from src.python.training.util import hashing, weightprob, Chronometer

import numpy as np
import pickle
import os


class Foldify:
    def __init__(self,
                 size,
                 nfolds=5,
                 train_prop=None,
                 val_prop=0.2,
                 sorted=True,
                 weight=None,
                 group=None,
                 seed=555,
                 datasets=None,
                 ftype='random'):
        size = int(size)
        nfolds = int(nfolds)
        train_prop = None if train_prop is None else float(train_prop)
        val_prop = float(val_prop)
        self.random = np.random.RandomState()
        if ftype == 'random':
            if group is not None:
                sub_group = group[np.arange(size)]
                nval = 0
                for lab in np.unique(sub_group):
                    nval += int(round(sum(sub_group == lab) * val_prop))
            else:
                nval = int(round(size * val_prop))
            train_size = [size - nval] * nfolds
            val_size = [nval] * nfolds
        elif ftype == 'rolling':
            weight, group = None, None
            ntrain = int(round(size * train_prop))
            nval = int(round(ntrain * val_prop))
            train_size = [ntrain] * nfolds
            val_size = [nval] * nfolds

        self.size = size
        self.nfolds = nfolds
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.ftype = ftype
        self.sorted = sorted
        self.seed = seed
        self.weight = weight
        self.group = group
        self.datasets = datasets
        self.train_size = train_size
        self.val_size = val_size

    @property
    def info(self):
        return dict(size=self.size,
                    nfolds=self.nfolds,
                    train_prop=self.train_prop,
                    val_prop=self.val_prop,
                    sorted=self.sorted,
                    seed=self.seed,
                    weight='\\non-added\\' if self.weight is None else '\\added\\',
                    group='\\non-added\\' if self.group is None else '\\added\\',
                    train_size=self.train_size,
                    val_size=self.val_size,
                    ftype=self.ftype)

    @property
    def hash(self):
        return hashing(self.info)

    def _random_index(self):
        index = np.arange(self.size)
        if self.group is None:
            weight = (None if self.weight is None
                      else weightprob(self.weight[index], abstolute=False))
            nval = int(round(self.size * self.val_prop))
            val_index = self.random.choice(
                index, size=nval, replace=False, p=weight)
        else:
            label = self.group[index]
            val_index = np.array([], dtype=int)
            for lab in np.unique(label):
                sub_index = index[label == lab]
                weight = (None if self.weight is None
                          else weightprob(self.weight[sub_index], abstolute=False))
                nval = int(round(len(sub_index) * self.val_prop))
                lab_val_index = self.random.choice(
                    sub_index, size=nval, replace=False, p=weight)
                val_index = np.concatenate((val_index, lab_val_index))
        val_index.sort()
        train = np.ones(self.size, dtype=bool)
        train[val_index] = False
        train_index = index[train]
        return train_index, val_index

    def _rolling_index(self, i):
        ntrain = self.train_size[0]
        nval = self.val_size[0]
        ind = int(i * (self.size-ntrain-nval) / self.nfolds)
        train_index = np.arange(ind, ind+ntrain)
        val_index = np.arange(ind+ntrain, ind+ntrain+nval)
        return train_index, val_index

    def get(self, i):
        self.random.seed(None if self.seed is None else self.seed + i)
        if self.ftype == 'random':
            train_index, val_index = self._random_index()
        elif self.ftype == 'rolling':
            train_index, val_index = self._rolling_index(i)

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

    def __str__(self):
        lst = ['Foldify:']
        for key, value in self.info.items():
            lst.append(f'{key}: {value}')

        return '\n\t'.join(lst)

    def Random(size,
               nfolds=5,
               val_prop=0.2,
               sorted=True,
               weight=None,
               group=None,
               seed=555,
               datasets=None):
        return Foldify(size=size,
                       nfolds=nfolds,
                       val_prop=val_prop,
                       sorted=sorted,
                       weight=weight,
                       group=group,
                       seed=seed,
                       datasets=datasets,
                       ftype='random')

    def Rolling(size,
                nfolds=5,
                train_prop=0.2,
                val_prop=0.2,
                sorted=True,
                seed=555,
                datasets=None):
        return Foldify(size=size,
                       nfolds=nfolds,
                       train_prop=train_prop,
                       val_prop=val_prop,
                       sorted=sorted,
                       weight=None,
                       group=None,
                       seed=seed,
                       datasets=datasets,
                       ftype='rolling')


def _format_summary(summary):

    return (f"\ttime: {int(summary['time'] // 3600)}h " +
            f"{int((summary['time'] % 3600) // 60)}m "
            f"{int(summary['time'] % 60)}s   " +
            '   '.join([f'{k}: {v:.6f}'
                        for k, v in summary.items() if k != 'time']))


def _total_history(history, train_size, val_size):
    total = {}
    for name in history[-1].keys():
        suma = 0
        size = val_size if name.startswith('val_') else train_size
        for i, hist in enumerate(history):
            suma += np.array(hist[name]) * size[i]
        total[name] = (suma / sum(size)).tolist()
    return total


def eval_model(model,
               hparam,
               foldify,
               infile=None,
               outfile=None,
               simplified=True,
               verbose=1) -> None:
    fit = model(**hparam)['fit']
    foldify_hash = foldify.hash
    if infile is None:
        session = {'foldify.hash': foldify_hash, 'data': dict()}
    elif isinstance(infile, dict):
        if len(infile) == 0:
            infile.update({'foldify.hash': foldify_hash, 'data': dict()})
        session = infile
    elif isinstance(infile, str):
        if os.path.exists(infile):
            with open(infile, "rb") as file:
                session = pickle.load(file)
        else:
            session = {'foldify.hash': foldify_hash, 'data': dict()}
            print(f"Session not found: '{infile}'.")
    else:
        raise ValueError(
            'Invalid input type for infile. Expected None, dict, or str.')

    if session['foldify.hash'] != foldify_hash:
        raise ValueError('Incompatible hash detected.')
    hparam_hash = hashing(hparam)

    new = not hparam_hash in session['data']
    if new:
        history, summary = [], []
    else:
        summary = session['data'][hparam_hash]['summary']
    if verbose in (1, 2, 3) and new:
        print('Hyperparameters:\n\t',
              '   '.join([f'{k}: {v}' for k, v in hparam.items()]))
    chrono = Chronometer(paused=True)
    for i in range(foldify.nfolds):

        if new:
            fold = foldify.get(i)
            chrono.start()
            fitted = fit(fold['train_set'], fold['val_set'],
                         verbose=verbose-1 if verbose in (2, 3) else 0)
            chrono.pause()
            history.append(fitted.history)
            summary.append({'time': chrono.partial(),
                            **{k: v[-1] for k, v in history[-1].items()}})
        if isinstance(summary, list) and verbose in (1, 2, 3):
            print(f'Fold {i+1}/{foldify.nfolds}:')
            print(_format_summary(summary[-1]))
    if new:
        history.append(
            _total_history(history, foldify.train_size, foldify.val_size))
        summary.append({'time': chrono.get(),
                        **{k: v[-1] for k, v in history[-1].items()}})
        session['data'][hparam_hash] = {'hparam': hparam,
                                        'history': history[-1] if simplified else history,
                                        'summary': summary[-1] if simplified else summary}
    if verbose in (1, 2, 3) and new:
        print('Total:')
        print(_format_summary(summary[-1]
                              if isinstance(summary, list) else summary))
    if outfile is not None:
        with open(outfile, "wb") as file:
            pickle.dump(session, file)
    return {'summary': summary[-1] if isinstance(summary, list) else summary,
            'session': session}
