from src.python.optimize.solution import Solution, ndSolution

from scipy.optimize import minimize
from typing import Iterable
import pandas as pd
import numpy as np
import pickle
import time
import math
import os


class Cooling:
    def __init__(self, fun, defaults, alpha=0.01, name=None, itermax=2, lag=1, tmax=1, tmin=0, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        names = list(set(defaults.keys())-set(kwargs.keys()))
        self.alpha = alpha
        self.kwargs = {k: v for k, v in kwargs.items() if k not in names}
        self.name = name
        self.itermax = itermax
        self.lag = lag
        self.tmax = tmax
        self.tmin = tmin
        if len(names) == 0:
            self.result = None
        else:
            x0 = [defaults[name] for name in names]

            def sqerror(x):
                return (fun(1, **{k: v for k, v in zip(names, x)}, **self.kwargs) - alpha) ** 2
            self.result = minimize(sqerror, x0=x0, method='Nelder-Mead')
            self.kwargs.update({k: v for k, v in zip(names, self.result['x'])})
        f0 = fun(0, **self.kwargs)
        f1 = fun(1, **self.kwargs)

        def function(x):
            if x < 0:
                return 1
            if x > 1:
                return 0
            if f1 != f0:
                return (f1 - fun(x, **self.kwargs)) / (f1 - f0)
            return 1 + fun(x, **self.kwargs) - f0
        self.function = function

    def temperature(self, iter, itermax=None, lag=None, tmax=None, tmin=None):
        itermax = self.itermax if itermax is None else itermax
        lag = self.lag if lag is None else lag
        tmax = self.tmax if tmax is None else tmax
        tmin = self.tmin if tmin is None else tmin
        return (tmin + (tmax - tmin) *
                self.function(((iter - 1) // lag) / ((itermax - 1) // lag)))

    def T(self, iter):
        return (self.tmin + (self.tmax - self.tmin) *
                self.function(((iter - 1) // self.lag) / ((self.itermax - 1) // self.lag)))

    def __str__(self) -> str:
        name = self.name if self.name is not None else ''
        args = ', '.join(
            [f'{k}={self.kwargs[k]}' for k in sorted(self.kwargs.keys())])
        args = '' if args == '' else ', '+args
        return f'{name}Cooling({self.alpha}{args})'

    def __repr__(self) -> str:
        return self.__str__()

    def Not(itermax=2, tmax=0, *_, **__):
        tmin = tmax
        def fun(x):
            if isinstance(x, Iterable):
                return np.ones(len(x), dtype=float)
            return 1
        return Cooling(fun, dict(), None, name='Not', itermax=itermax, tmax=tmax, tmin=tmin)

    def Lin(itermax=2, lag=1, tmax=1, tmin=0, *_, **__):
        return Cooling(lambda x: 1 - x, dict(), None, name='Lin',
                       itermax=itermax, lag=lag, tmax=tmax, tmin=tmin)

    def Exp(alpha, itermax=2, lag=1, tmax=1, tmin=0, cte=None, shape=0.0, *_, **__):
        def fun(x, cte, shape):
            return (1 + shape * 100) / (shape * 100 + cte ** (-x))
        defaults = dict(cte=0.01, shape=0.0)
        return Cooling(fun, defaults, alpha, cte=cte, shape=shape, name='Exp',
                       itermax=itermax, lag=lag, tmax=tmax, tmin=tmin)

    def Slow(alpha, itermax=2, lag=1, tmax=1, tmin=0, cte=None, shape=0.0, *_, **__):
        def fun(x, cte, shape):
            return (1 + shape * 100) / (1 + shape * 100 + x / cte)
        defaults = dict(cte=0.01, shape=0.0)
        return Cooling(fun, defaults, alpha, cte=cte, shape=shape, name='Slow',
                       itermax=itermax, lag=lag, tmax=tmax, tmin=tmin)


class SearchSpace:
    def __init__(self, solution, seed=None) -> None:
        self.solution = solution
        if seed is not None:
            self.seed = seed

    @property
    def seed(self):
        raise AttributeError('The "seed" value is not available')

    @seed.setter
    def seed(self, seed):
        self.solution.seed = seed

    @property
    def random(self):
        return self.solution.random

    def is_feasible(self, _):
        return True

    def new(self):
        x = self.solution.new()
        while not self.is_feasible(x):
            x = self.solution.new()
        return x

    def cross(self, x, y, **kwargs):
        z = self.solution.do('cross', x, y, **kwargs)
        while not self.is_feasible(z):
            z = self.solution.do('cross', x, y, **kwargs)
        return z

    def mutate(self, x, **kwargs):
        y = self.solution.do('mutate', x, **kwargs)
        while not self.is_feasible(y):
            y = self.solution.do('mutate', x, **kwargs)
        return y

    def neighbor(self, x, **kwargs):
        if isinstance(self.solution, ndSolution) and len(x) > 1:
            x = x.copy()
            i = self.random.randint(len(x))
            x[i] = self.solution[i].do('neighbor', x[i], **kwargs)
            return x
        return self.solution.do('neighbor', x, **kwargs)

    def __repr__(self) -> str:
        return f'SearchSpace({self.solution})'

    def __str__(self) -> str:
        return f'SearchSpace({self.solution})'


class Chronometer:
    def __init__(self, initial=0, paused=False):
        self.start(initial)
        if paused:
            self.pause()

    def start(self, initial=0):
        self.start_time = time.time()-initial
        self.paused_time = 0
        self.paused = False
        self.paused_start_time = None

    def get(self):
        current_time = time.time()
        if self.paused:
            lapsed_time = self.paused_start_time - self.start_time - self.paused_time
        else:
            lapsed_time = current_time - self.start_time - self.paused_time
        return lapsed_time

    def pause(self):
        if not self.paused:
            self.paused_start_time = time.time()
            self.paused = True

    def release(self):
        if self.paused:
            self.paused_time += time.time() - self.paused_start_time
            self.paused = False


class Search():
    def __init__(self, function,
                 search_space,
                 initial=None,
                 itermax=None,
                 evalmax=None,
                 timemax=None,
                 infile=None,
                 outfile=None,
                 seed=None,
                 argstype=None,
                 verbose=0) -> None:
        self.verbose = verbose
        self.ehistory = []
        self.ihistory = []
        self.frozen = False
        if infile is not None and os.path.exists(infile):
            if isinstance(infile, str):
                session = pickle.load(open(infile, "rb"))
                print(f"Session loaded from '{infile}'.")
            else:
                session = infile
            if session.niter > 0 and session.ehistory is not None:
                self.ehistory = self.unpack_ehistory(session.ehistory)
                if session.ihistory is not None:
                    self.ihistory = self.unpack_ihistory(session.ihistory)
                self.frozen = True
        self.search_space = (SearchSpace(search_space)
                             if isinstance(search_space, Solution) else search_space)
        self.solution = self.search_space.solution
        if seed is not None:
            self.search_space.seed = seed

        if argstype == 'args':
            if not isinstance(self.solution, ndSolution):
                raise ValueError(
                    "argstype='args' is only avaiable for ndSolution instances.")
            self.function = lambda x: function(*self.solution.decode(x))
        elif argstype == 'kwargs':
            self.function = lambda x: function(
                *self.solution.decode(x, to_dict=True))
        elif argstype == 'literal':
            self.function = lambda x: function(x)
        else:
            self.function = lambda x: function(self.solution.decode(x))

        inf = float('Inf')
        self.itermax = inf if itermax is None else itermax
        self.evalmax = inf if evalmax is None else evalmax
        self.timemax = inf if timemax is None else timemax
        if self.frozen:
            self.itermax += session.niter
            self.evalmax += session.neval
            self.timemax += session.time
        self.niter = 0
        self.neval = 0
        self.chrono = Chronometer(paused=self.frozen)
        self.xmin = None
        self.fmin = inf
        if initial is not None:
            self.eval(self.solution.read(initial))
        self.begin()
        if self.itermax < inf or self.evalmax < inf or self.timemax < inf:
            while not (self.niter >= self.itermax or
                       self.neval >= self.evalmax or
                       self.chrono.get() >= self.timemax):
                self.niter += 1
                self.kernel()
                if self.frozen:
                    if self.niter + 1 == session.niter:
                        self.frozen = False
                        self.chrono.start(session.time)
                else:
                    self.iter_saving()
        self.chrono.pause()
        if outfile is not None:
            pickle.dump(self.session, open(outfile, "wb"))
            print(f"Session saved to '{outfile}'.")

    @property
    def random(self):
        return self.solution.random

    def unpack_ehistory(self, history):
        return history.to_dict(orient='records')

    def unpack_ihistory(self, history):
        return history.to_dict(orient='records')

    def pack_ehistory(self, history):
        return (None if len(history) == 0
                else pd.DataFrame(history))

    def pack_ihistory(self, history):
        return (None if len(history) == 0
                else pd.DataFrame(history))

    def __str__(self) -> str:
        return 'session:\n' + str(self.session)

    def __repr__(self) -> str:
        return 'session:\n' + str(self.session)

    @property
    def session(self):
        return pd.Series(dict(time=self.chrono.get(),
                              niter=self.niter,
                              neval=self.neval,
                              fmin=self.fmin,
                              xmin=(None if self.xmin is None
                                    else self.solution.write(self.xmin)),
                              ehistory=self.pack_ehistory(self.ehistory),
                              ihistory=self.pack_ihistory(self.ihistory)
                              ), dtype=object)

    @property
    def verbosing(self):
        return ('\nProgress:'
                f'\n\titer: {self.niter}/{self.itermax}'
                f'\n\teval: {self.neval}/{self.evalmax}'
                f'\n\ttime: {round(self.chrono.get(), 3)}/{self.timemax}'
                f'\n\tfmin: {self.fmin}'
                f'\n\txmin: {self.solution.decode(self.xmin)}')

    def eval_saving(self):
        if not self.frozen:
            self.chrono.pause()
            eval_dict = self.eval_dict
            if eval_dict is not None:
                self.ehistory.append(eval_dict)
            if self.verbose == 1:
                print(self.verbosing)
            self.chrono.release()

    def iter_saving(self):
        if not self.frozen:
            self.chrono.pause()
            iter_dict = self.iter_dict
            if iter_dict is not None:
                self.ihistory.append(iter_dict)
            if self.verbose == 2:
                print(self.verbosing)
            self.chrono.release()

    def Fx(self):
        if self.frozen:
            if self.ehistory[self.neval - 1]['x'] != self.solution.write(self.x):
                raise ValueError("Something went wrong with the generation of random solutions.\n"
                                 "Possible causes:\n"
                                 "\t- Initial conditions have been changed\n"
                                 "\t- The search space has been modified\n"
                                 "\t- A different seed has been used")
            return self.ehistory[self.neval - 1]['fx']
        return self.function(self.x)

    def eval(self, x):
        self.neval += 1
        self.x = x
        self.fx = self.Fx()
        if self.fx <= self.fmin:
            self.xmin = self.x
            self.fmin = self.fx
        self.eval_saving()
        return self.fx

    @property
    def eval_dict(self):
        return dict(time=self.chrono.get(),
                    niter=self.niter,
                    fmin=self.fmin,
                    fx=self.fx,
                    x=self.solution.write(self.x))

    @property
    def iter_dict(self):
        return None

    def kernel(self):
        pass

    def begin(self):
        pass


class RandomSearch(Search):
    def kernel(self):
        self.eval(self.search_space.new())


class LocalSearch(Search):

    def __init__(self, function, search_space, initial=None, cooling=None, itermax=None, evalmax=None, timemax=None, infile=None, outfile=None, seed=None, argstype=None, verbose=0) -> None:
        self.xact = None
        self.fact = float('inf')
        cooling = Cooling.Not(tmax=0) if cooling is None else cooling
        self.Tk = lambda: cooling.T(self.niter)
        super().__init__(function, search_space, initial, itermax, evalmax, timemax,
                         infile, outfile, seed, argstype, verbose)

    def eval(self, x):
        self.neval += 1
        self.x = x
        self.fx = self.Fx()
        self.T = self.Tk()
        if self.fx <= self.fact:
            self.xact = self.x
            self.fact = self.fx
            if self.fx <= self.fmin:
                self.xmin = self.x
                self.fmin = self.fx
        elif (self.T > 0 and
              self.search_space.random.rand() <
              math.exp(-(self.fx - self.fact) / self.T)):
            self.xact = self.x
            self.fact = self.fx
        self.eval_saving()
        return self.fx

    @property
    def eval_dict(self):
        return dict(time=self.chrono.get(),
                    niter=self.niter,
                    T=self.T,
                    fact=self.fact,
                    fmin=self.fmin,
                    fx=self.fx,
                    x=self.solution.write(self.x))

    def kernel(self):
        if self.xact is None:
            self.eval(self.search_space.new())
        else:
            self.eval(self.search_space.neighbor(self.xact))


def weightprob(weight):
    weight = np.array(weight, dtype=float)
    weight -= weight.min()
    n = len(weight)
    y = weight.sum()
    if y == 0:
        return np.full(n, 1 / n)
    p = 1 / (n * 10 ** 6)
    weight += p * y / (1 - p * n)
    return weight / weight.sum()


def roulette(data, weight=None, size=1, k=1, replace=True, random=None):
    if random is None:
        random = np.random
    if weight is None:
        weight = np.full(len(weight), 1)
    df = None
    if isinstance(data, pd.DataFrame):
        df = data
        data = np.arange(len(data))

    def roulette1(data, weight, size, replace):
        return random.choice(data, size=size, replace=replace, p=weightprob(weight))

    if k == 1:
        result = roulette1(data, weight, size, replace)
    else:
        def rouletteK(data, weight, size, replace):
            n = len(data)
            basis = roulette1(data=np.arange(n), weight=weight,
                              size=int(np.ceil(size / k)), replace=replace)
            sample = [basis]
            for i in range(1, k):
                sample.append([int((basis[j] + i * n / k) % n)
                               for j in range(len(basis))])
            return data[np.concatenate(sample)]
        if replace:
            result = rouletteK(data, weight, size, True)[:size]
        else:
            locs = np.ones(len(data), dtype=bool)
            locsize = 0
            while locsize < size:
                locs[rouletteK(np.arange(len(data))[locs], weight[locs],
                               size - locsize, False)] = False
                locsize = len(data) - sum(locs)
            result = data[~locs][:size]
    if df is None:
        return result
    df = df.iloc[result]
    df.reset_index(inplace=True, drop=True)
    return df


def rank_selection(data, weight=None, s=0.75, size=1, replace=True, random=None):
    if random is None:
        random = np.random
    if weight is None:
        weight = np.full(len(weight), 1)
    df = None
    if isinstance(data, pd.DataFrame):
        df = data
        data = np.arange(len(data))
    i = np.argsort(data)
    n = len(data)
    prob = 2*(1-s)/n + 2*i*(2*s-1)/(n**2-n)
    result = random.choice(data, size=size, replace=replace, p=prob)
    if df is None:
        return result
    df = df.iloc[result]
    df.reset_index(inplace=True, drop=True)
    return df


def tournament(data1, weight1, data2=None, weight2=None, size=None, replace=True, dim=2, random=None):
    isdf = False
    if isinstance(data1, pd.DataFrame):
        isdf = True
        df1 = data1
        data1 = np.arange(len(data1))
        if data2 is not None:
            df2 = data2
            data2 = np.arange(len(data2))
    weight1 = np.array(weight1)
    if weight2 is not None:
        weight2 = np.array(weight2)

    random = np.random if random is None else random
    if data2 is None:
        size = len(data1) // dim if size is None else size
        result = []
        index1 = random.choice(np.arange(len(data1)),
                               size=size * dim, replace=replace)
        for i in range(size):
            ind = np.arange(dim * i, dim * (i + 1))
            weightmax = -float('Inf')
            datamax = None
            for j in index1[ind]:
                if weight1[j] > weightmax:
                    weightmax = weight1[j]
                    datamax = data1[j]
            result.append(datamax)

        return df1.iloc[result] if isdf else np.array(result)

    dim = int(np.ceil(dim / 2))
    size = min(len(data1), len(data2)) // dim if size is None else size
    result1, result2 = [], []
    index1 = random.choice(np.arange(len(data1)),
                           size=size * dim, replace=replace)
    index2 = random.choice(np.arange(len(data2)),
                           size=size * dim, replace=replace)
    for i in range(size):
        ind = np.arange(dim * i, dim * (i + 1))
        weightmax = -float('Inf')
        datamax = None
        for result, data, weight, index in ((result1, data1, weight1, index1[ind]),
                                            (result2, data2, weight2, index2[ind])):
            for j in index:
                if weight[j] > weightmax:
                    weightmax = weight[j]
                    datamax = result, data[j]
        datamax[0].append(datamax[1])
    if isdf:
        df = pd.concat([df1.iloc[result1], df2.iloc[result2]], axis=0)
        df.reset_index(inplace=True, drop=True)
        return df
    return np.concatenate([result1, result2])


class GeneticSearch(Search):
    def __init__(self, function, search_space, initial=None, itermax=None, evalmax=None, timemax=None, infile=None, outfile=None, seed=None, argstype=None, verbose=0) -> None:
        if isinstance(initial, (int, float)):
            initial = RandomSearch(function, search_space, itermax=initial,
                                   seed=seed, argstype=argstype).session.ehistory
        if not isinstance(initial, pd.DataFrame):
            raise ValueError('Initial value must be a pd.Dataframe instance')
        search_space = (SearchSpace(search_space)
                        if isinstance(search_space, Solution) else search_space)
        self.population = pd.DataFrame(dict(fx=initial['fx'],
                                            x=[search_space.solution.read(x) for x in initial['x']]))
        super().__init__(function, search_space, None, itermax, evalmax, timemax,
                         infile, outfile, seed, argstype, verbose)

    def unpack_ihistory(self, history):
        group = history.groupby('niter')
        history = pd.DataFrame({name: group[name].apply(list)
                                if name in ('x', 'fx') else group[name].first()
                                for name in history.columns})
        return history.to_dict(orient='records')

    def pack_ihistory(self, history):
        return (None if len(history) == 0
                else pd.concat([pd.DataFrame(x)
                                for x in history], axis=0))

    def begin(self):
        self.iter_saving()

    @property
    def iter_dict(self):
        return dict(time=self.chrono.get(),
                    niter=self.niter,
                    neval=self.neval,
                    fx=self.population.fx.to_list(),
                    x=[self.solution.write(x) for x in self.population.x])

    def kernel(self):
        parents = self.selection()
        solutions = self.reproduction(parents)
        childs = self.eval_solutions(solutions)
        newpop = self.replacement(parents, childs)
        self.population = self.elitism(newpop, parents, childs)

    def selection(self):
        return roulette(self.population,
                        weight=-self.population.fx,
                        size=len(self.population)//2,
                        replace=True,
                        random=self.random)

    def reproduction(self, parents):
        solutions = []
        for _ in range(len(self.population)):
            choised = self.random.choice(parents.x, size=2, replace=False)
            new = self.search_space.cross(*choised)
            new = self.search_space.mutate(new)
            solutions.append(new)
        return solutions

    def eval_solutions(self, solutions):
        return pd.DataFrame(dict(fx=[self.eval(sol) for sol in solutions], x=solutions))

    def replacement(self, parents, childs):
        return tournament(self.population,
                          -self.population.fx,
                          childs,
                          -childs.fx,
                          size=len(self.population),
                          replace=False,
                          random=self.random)

    def elitism(self, newpop, parents, childs):
        if self.fmin < min(newpop.fx):
            index = newpop.fx.idxmax()
            newpop.at[index, 'fx'] = self.fmin
            newpop.at[index, 'x'] = self.xmin
        return newpop
