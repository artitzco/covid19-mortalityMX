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
        x = self.solution.cross(x, y, **kwargs)
        while not self.is_feasible(x):
            x = self.solution.cross(x, y, **kwargs)
        return x

    def mutate(self, x, **kwargs):
        x = self.solution.mutate(x, **kwargs)
        while not self.is_feasible(x):
            x = self.solution.mutate(x, **kwargs)
        return x

    def neighbor(self, x, **kwargs):
        x = self.solution.neighbor(x, **kwargs)
        while not self.is_feasible(x):
            x = self.solution.neighbor(x, **kwargs)
        return x

    def __repr__(self) -> str:
        return f'SearchSpace({self.solution})'

    def __str__(self) -> str:
        return f'SearchSpace({self.solution})'


class Chronometer:

    def __init__(self, initial=0, paused=False):
        current_time = time.time()
        self.start_time = current_time - initial
        self.paused_time = 0
        self.paused = paused
        self.paused = paused
        self.paused_initial = current_time if paused else None
        self.last = initial

    def start(self, initial=0):
        if self.paused:
            self.paused_time += time.time() - self.paused_initial
            self.paused = False
        else:
            self.start_time = time.time() - initial
            self.paused_time = 0
            self.paused = False
            self.paused_initial = None

    def get(self):
        if self.paused:
            result = self.paused_initial - self.start_time - self.paused_time
        else:
            result = time.time() - self.start_time - self.paused_time
        self.last = result
        return result

    def pause(self):
        if not self.paused:
            self.paused_initial = time.time()
            self.paused = True

    def partial(self):
        last = self.last
        return self.get() - last


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
        self.history = []
        self.iter_history = []
        self.frozen = False
        if infile is not None:
            if os.path.exists(infile):
                if isinstance(infile, str):
                    session = pickle.load(open(infile, "rb"))
                    print(f"Session loaded from '{infile}'.")
                else:
                    session = infile
                if session.niter > 0 and session.history is not None:
                    self.history = session.history.to_dict(orient='records')
                    self.frozen = True
                if session.iter_history is not None:
                    self.iter_history = session.iter_history.to_dict(
                        orient='records')
            else:
                print(f"Session not found:'{infile}'.")
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
                **self.solution.decode(x, to_dict=True))
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
        self.chrono = Chronometer()
        self.xmin = None
        self.fmin = inf
        if initial is not None:
            self.eval(self.solution.read(initial))
        self.begin()

        if self.itermax < inf or self.evalmax < inf or self.timemax < inf:
            while not (
                    self.niter >= self.itermax or
                    self.neval >= self.evalmax or
                    self.chrono.get() >= self.timemax):
                self.niter += 1
                self.kernel()
                if self.frozen:
                    if self.niter == session.niter:
                        self.frozen = False
                        self.chrono = Chronometer(initial=session.time)
                else:
                    self.iter_saving()
        self.chrono.pause()
        if outfile is not None:
            pickle.dump(self.session, open(outfile, "wb"))
            print(f"Session saved to '{outfile}'.")

    @property
    def random(self):
        return self.solution.random

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
                              history=(None if len(self.history) == 0
                                       else pd.DataFrame(self.history)),
                              iter_history=(None if len(self.iter_history) == 0
                                            else pd.DataFrame(self.iter_history))
                              ), dtype=object)

    @property
    def verbosing(self):
        return ('Progress:'
                f'\n\titer: {self.niter}/{self.itermax}'
                f'\n\teval: {self.neval}/{self.evalmax}'
                f'\n\ttime: {round(self.chrono.get(), 3)}/{None if self.timemax is None else  round(self.timemax, 3)}'
                f'\n\tfmin: {self.fmin}'
                f'\n\txmin: {None if self.xmin is None else self.solution.decode(self.xmin)}')

    def eval_saving(self):
        if not self.frozen:
            self.chrono.pause()
            eval_dict = self.eval_dict
            if eval_dict is not None:
                self.history.append(eval_dict)
            if self.verbose in [1, 3]:
                print(self.verbosing)
            self.chrono.start()

    def iter_saving(self):
        if not self.frozen:
            self.chrono.pause()
            iter_dict = self.iter_dict
            if iter_dict is not None:
                self.iter_history.append(iter_dict)
            if self.verbose == [2, 3]:
                print(self.verbosing)
            self.chrono.start()

    def Fx(self):
        if self.frozen:
            if self.history[self.neval - 1]['x'] != self.solution.write(self.x):
                raise ValueError("Something went wrong with the generation of random solutions.\n"
                                 "Possible causes:\n"
                                 "\t- Initial conditions have been changed\n"
                                 "\t- The search space has been modified\n"
                                 "\t- A different seed has been used")

            return self.history[self.neval - 1]['fx']
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


def weightprob(weight, abstolute=False):
    weight = np.array(weight, dtype=float)
    if abstolute:
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
        return random.choice(data, size=size, replace=replace,
                             p=weightprob(weight, abstolute=True))

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
                                   seed=seed, argstype=argstype).session.history
        if not isinstance(initial, pd.DataFrame):
            raise ValueError('Initial value must be a pd.Dataframe instance')
        search_space = (SearchSpace(search_space)
                        if isinstance(search_space, Solution) else search_space)
        self.population = pd.DataFrame(dict(fx=initial['fx'],
                                            x=[search_space.solution.read(x) for x in initial['x']]))
        super().__init__(function, search_space, None, itermax, evalmax, timemax,
                         infile, outfile, seed, argstype, verbose)

    def begin(self):
        self.iter_saving()

    @property
    def iter_dict(self):
        return dict(time=self.chrono.get(),
                    niter=self.niter,
                    neval=self.neval,
                    fmin=self.population.fx.min(),
                    fmean=self.population.fx.mean(),
                    fstd=self.population.fx.std())

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
        popsize = len(self.population)
        for _ in range(popsize):
            choised = self.random.choice(parents.x, size=2, replace=False)
            new = self.search_space.cross(*choised)
            dim = popsize * self.search_space.solution.dimension(new)
            new = self.search_space.mutate(new, prob=1/dim)
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


def entropy_model(probability, initial_entropy, a=0, b=1, c=1):
    if hasattr(probability, '__iter__'):
        return np.array([entropy_model(p, initial_entropy, a, b, c) for p in probability])
    d = b + (initial_entropy-1) * (b - a)
    return c + (d-c) * (1-probability)


def entropy_model_inv(final_entropy, initial_entropy, a=0, b=1, c=1):
    if hasattr(final_entropy, '__iter__'):
        return np.array([entropy_model_inv(e, initial_entropy, a, b, c) for e in final_entropy])
    d = b + (initial_entropy-1) * (b - a)
    if c < d:
        if final_entropy <= c:
            return 1
        if final_entropy >= d:
            return 0
    else:
        if final_entropy >= c:
            return 1
        if final_entropy <= d:
            return 0
    return 1 - (final_entropy-c) / (d-c)


class MarkovGeneticSearch(GeneticSearch):

    def __init__(self, function, search_space, entropy_limit=0.05, initial=None, itermax=None,
                 evalmax=None, timemax=None, infile=None, outfile=None, seed=None, argstype=None, verbose=0) -> None:
        self.entropy_limit = entropy_limit
        super().__init__(function, search_space, initial, itermax,
                         evalmax, timemax, infile, outfile, seed, argstype, verbose)

    def begin(self):
        if len(self.iter_history) > 0:
            self.initial_entropy = [hist['initial_entropy']
                                    for hist in self.iter_history]
            self.final_entropy = [hist['final_entropy']
                                  for hist in self.iter_history]
            self.probability = [hist['probability']
                                for hist in self.iter_history]
        else:
            self.initial_entropy = []
            self.final_entropy = []
            self.probability = [float('nan')]
            self.iter_saving()

    @property
    def iter_dict(self):
        self.initial_entropy.append(
            self.final_entropy[-1] if len(self.final_entropy) > 0 else float('nan'))
        self.final_entropy.append(
            self.solution.entropy(self.population.x.tolist()))
        return dict(time=self.chrono.get(),
                    niter=self.niter,
                    neval=self.neval,
                    fmin=self.population.fx.min(),
                    fmean=self.population.fx.mean(),
                    fstd=self.population.fx.std(),
                    initial_entropy=self.initial_entropy[-1],
                    final_entropy=self.final_entropy[-1],
                    probability=self.probability[-1])

    def selection(self):
        return self.population

    def reproduction(self, parents):
        entropy = self.solution.entropy(self.population.x.tolist())
        solutions = []
        popsize = len(self.population)

        case_a = entropy <= self.entropy_limit
        case_b = entropy > self.entropy_limit
        if case_a:
            prob = entropy_model_inv(
                final_entropy=self.entropy_limit,
                initial_entropy=entropy,
                a=0.0,
                b=1.0,
                c=0.95)
        else:
            mean_prob = 0
        for _ in range(popsize):
            choised = self.random.choice(parents.x, size=2, replace=False)
            new = self.search_space.cross(*choised)
            if case_b:
                dim = popsize * self.search_space.solution.dimension(new)
                prob = - math.log(self.random.rand()) / dim
                prob = min(prob, 1)
                mean_prob += prob
            new = self.search_space.mutate(new, prob=prob)
            solutions.append(new)
        self.probability.append(mean_prob / popsize
                                if case_b else prob)
        return solutions

    def replacement(self, _, childs):
        population = pd.concat([self.population, childs], axis=0)
        population.reset_index(drop=True, inplace=True)
        return tournament(population,
                          -population.fx,
                          size=len(self.population),
                          replace=False,
                          random=self.random)
