from src.python.optimize.binary import ndBits, Bits, Binary, ndBinary
from typing import Callable, Iterable, Any
from scipy.optimize import minimize
import os.path as path
import pandas as pd
import pickle
import math
import json
import time
import sys


class Cooling:
    def __init__(self, fun, defaults, itermax, lag=1, tmax=1, tmin=0.01, **kwargs):
        names = list(set(defaults.keys())-set(kwargs.keys()))
        self.itermax = itermax
        self.lag = lag
        self.tmin = tmin
        self.tmax = tmax
        self.kwargs = {k: v for k, v in kwargs.items() if k not in names}
        if len(names) == 0:
            self.result = None
        else:
            x0 = [defaults[name] for name in names]

            def sqerror(x):
                return (tmax * fun(1, **{k: v for k, v in zip(names, x)}, **self.kwargs) - tmin) ** 2
            self.result = minimize(sqerror, x0=x0)
            self.kwargs.update({k: v for k, v in zip(names, self.result['x'])})

        def temperature(x):
            return tmax * fun(x, **self.kwargs)
        self.temperature = temperature

    @property
    def temperatureK(self):
        return lambda iter: self.temperature(((iter - 1) // self.lag) / ((self.itermax - 1) // self.lag))


class NoneCooling(Cooling):
    def __init__(self, t=0):
        super().__init__(lambda _: 1, dict(), float('nan'), t, t, t)

    @property
    def temperatureK(self):
        return lambda _: self.tmax


class ExpCooling(Cooling):
    def __init__(self, itermax, lag=1, tmax=1, tmin=0.01, defaults=dict(cte=0.01, shape=0.0), **kwargs):
        def fun(x, cte, shape):
            return (1 + shape * 100) / (shape * 100 + cte ** (-x))
        super().__init__(fun, defaults, itermax, lag, tmax, tmin, **kwargs)


class SlowCooling(Cooling):
    def __init__(self, itermax, lag=1, tmax=1, tmin=0.01, defaults=dict(cte=0.01, shape=0.0), **kwargs):
        def fun(x, cte, shape):
            return (1 + shape * 100) / (1 + shape * 100 + x / cte)
        super().__init__(fun, defaults, itermax, lag, tmax, tmin, **kwargs)


class Function:
    def __init__(self,
                 objetive: Callable,
                 argstype: str = None,
                 **kwargs):
        if not callable(objetive):
            raise ValueError(f'objetive must be a function')
        types = [None, 'args', 'kwargs']
        if argstype not in types:
            raise ValueError(f'argstype must be in {types}')

        self.objetive = objetive
        self.argstype = argstype
        self.kwargs = kwargs

    def eval(self, bits: ndBits | Bits):
        if self.argstype == 'args':
            return self.objetive(*bits.value, **self.kwargs)
        if self.argstype == 'kwargs':
            return self.objetive(**bits.value_dict, **self.kwargs)
        return self.objetive(bits.value, **self.kwargs)


class SearchSpace:
    def __init__(self, binary: Iterable[Binary] | Binary | ndBinary,
                 seed: int = None):
        if hasattr(binary, '__iter__'):
            if all(isinstance(bin, Binary) for bin in binary):
                self.binary = ndBinary(*binary)
            else:
                raise ValueError(
                    'If `binary` is iterable, it must only contain instances of the `Binary` class')
        elif isinstance(binary, ndBinary) or isinstance(binary, Binary):
            self.binary = binary
        else:
            raise ValueError(
                'If `binary` is not iterable, it must be an instance of the `Binary` or `ndBinary` class')
        self.seed = seed

    @property
    def random(self):
        return self.binary.random

    @property
    def seed(self):
        raise AttributeError("seed value not available.")

    @seed.setter
    def seed(self, seed):
        self.binary.seed = seed

    def is_feasible(self, _: ndBits | Bits):
        return True

    def neighbor(self, bits: ndBits | Bits, probability: float = None, epsilon: int = 1):
        if probability is None:
            probability = 1.0 / self.binary.dimension
        count, newbits = bits.mutate(probability, inplace=False)
        return (newbits if (self.is_feasible(newbits)
                and count > 0 and count <= epsilon)
                else self.neighbor(bits, probability, epsilon))

    def mutation(self, bits: ndBits | Bits, probability: float = None):
        if probability is None:
            probability = 1.0 / self.binary.dimension
        _, newbits = bits.mutate(probability, inplace=False)
        return (newbits if self.is_feasible(newbits)
                else self.mutation(bits, probability))

    def crossing(self, bits1: ndBits | Bits, bits2: ndBits | Bits, probability: float | Iterable[float] = 0.5):
        if not isinstance(probability, float):
            probability = self.random.uniform(probability[0], probability[1])
        _, newbits = bits1.cross(bits2, probability, inplace=False)
        return (newbits if self.is_feasible(newbits)
                else self.crossing(bits1, bits2, probability))

    def new(self, x=None):
        bits = self.binary.new(x)
        if self.is_feasible(bits):
            return bits
        if x is None:
            return self.new()
        raise ValueError('You are trying to create a non-feasible solution!')


class RandomSearch:
    def __init__(self,
                 function: Function,
                 search_space: SearchSpace | Iterable[Binary] | Binary | ndBinary,
                 initialize: ndBits | Bits | bool | Any = True,
                 seed: int = None,
                 saving: bool = True):

        self.function = function
        self.search_space = (search_space if isinstance(search_space, SearchSpace)
                             else SearchSpace(search_space))

        initial = None if isinstance(initialize, bool) else initialize
        self.initial = (None if initial is None else
                        (initial.bits if isinstance(initial, ndBits | Bits) else initial))
        self.initialize = initialize if isinstance(initialize, bool) else True
        self.seed = seed
        self.saving = saving

    def eval(self, x):
        if (self.neval < len(self.history) and
                x.bits != self.history[self.neval]['x']):
            raise ValueError('Unexpected value of x!\n'
                             f'Error occurred in evaluation {self.neval}, '
                             f'iteration {self.niter}, and time {time.time()-self.seconds}\n'
                             'Possible causes:\n' '1 Initial conditions have been modified\n'
                             '2 The random number generation seed is "None"\n'
                             '3 Unexpected error in random number generation')

        self.x = x
        self.f = (self.history[self.neval]['f']
                  if self.neval < len(self.history)
                  else self.function.eval(x))
        if self.f < self.fmin:
            self.xmin = x
            self.fmin = self.f
        return self.f

    def save(self):
        if self.saving:
            if self.neval >= len(self.history):
                self.history.append(self.history_item)
            if self.neval == len(self.history) - 1:
                self.seconds = time.time() - self.history[self.neval]['time']
        self.neval += 1

    @property
    def history_item(self):
        return dict(iter=self.niter,
                    eval=self.neval,
                    time=time.time() - self.seconds,
                    fmin=self.fmin,
                    f=self.f,
                    x=self.x.bits)

    @property
    def session(self):
        return pd.Series(dict(iter=self.niter,
                              eval=self.neval,
                              time=time.time() - self.seconds,
                              fmin=self.fmin,
                              xmin=None if self.xmin is None else self.xmin.bits,
                              history=pd.DataFrame(self.history)))

    def begin(self): pass

    def kernel(self):
        self.eval(self.search_space.new())
        self.save()

    def verbosing(self, verbose, itermax, evalmax, timemax):
        if verbose in [1, 2, 3]:

            # 1 bits
            # 2 value
            # 3 value_dict
            xmin = (None if self.xmin is None
                    else (self.xmin.bits if verbose == 1
                          else (self.xmin.value if verbose == 2
                                else self.xmin.value_dict)))
            session_time = round(time.time() - self.seconds, 3)
            print('\nSession = ',
                  json.dumps(dict(iter=f'{self.niter} / {itermax if itermax < sys.maxsize else None}',
                                  eval=f'{self.neval} / {evalmax if evalmax < sys.maxsize else None}',
                                  time=f'{session_time} / {timemax}',
                                  fmin=self.fmin,
                                  xmin=xmin), indent=4), '\n')

    def run(self,
            itermax: int = None,
            evalmax: int = None,
            timemax: float = None,
            session: dict = None,
            infile: str = None,
            outfile: str = None,
            verbose: int = 0):

        if infile is not None and path.exists(infile):
            with open(infile, "rb") as file:
                session = pickle.load(file)

        if timemax is None and itermax is None and evalmax is None:
            itermax, evalmax = 0, 0
        else:
            itermax = sys.maxsize if itermax is None else itermax
            evalmax = sys.maxsize if evalmax is None else evalmax
        timemax = timemax

        if session is None or not self.saving:
            self.history = []
        else:
            self.history = session.history.to_dict(orient='records')
            if itermax < sys.maxsize:
                itermax += session['iter']
            if evalmax < sys.maxsize:
                evalmax += session['eval']
            if timemax is not None:
                timemax += session['time']

        self.search_space.seed = self.seed
        self.x = None
        self.f = None
        self.xmin = None
        self.fmin = float('inf')
        self.niter = 0
        self.neval = 0
        self.seconds = time.time()
        if self.initialize:
            self.eval(self.search_space.new(self.initial))
            self.save()
        self.begin()
        for self.niter in range(1, itermax + 1):
            if self.neval >= evalmax:
                break
            if (timemax is not None
                    and time.time() - self.seconds >= timemax):
                break
            self.kernel()
            if session is None or self.niter > session['iter']:
                if outfile is not None:
                    with open(outfile, "wb") as file:
                        pickle.dump(self.session, file)
                self.verbosing(verbose, itermax, evalmax, timemax)

        return self.session


def random_search(function: Function,
                  search_space: SearchSpace | Iterable[Binary] | Binary | ndBinary,
                  initialize: ndBits | Bits | bool | Any = True,
                  seed: int = None,
                  saving: bool = True,
                  itermax: int = None,
                  evalmax: int = None,
                  timemax: float = None,
                  session: dict = None,
                  infile: str = None,
                  outfile: str = None,
                  verbose: int = 0):
    return RandomSearch(function=function,
                        search_space=search_space,
                        initialize=initialize,
                        seed=seed,
                        saving=saving).run(itermax=itermax,
                                           evalmax=evalmax,
                                           timemax=timemax,
                                           session=session,
                                           infile=infile,
                                           outfile=outfile,
                                           verbose=verbose)


class HillClimbing(RandomSearch):

    def __init__(self, function: Function,
                 search_space: SearchSpace | Iterable[Binary] | Binary | ndBinary,
                 initialize: ndBits | Bits | bool | Any = None,
                 seed: int = None,
                 saving: bool = True):
        super().__init__(function, search_space, initialize, seed, saving)
        if isinstance(initialize, bool) and not initialize:
            raise ValueError('The hill climbing algorithm must be initialized')

    def kernel(self):
        for neighbor in self.xmin.neighborhood():
            self.eval(neighbor)
            self.save()


def hill_climbing(function: Function,
                  search_space: SearchSpace | Iterable[Binary] | Binary | ndBinary,
                  initialize: ndBits | Bits | bool | Any = None,
                  seed: int = None,
                  saving: bool = True,
                  itermax: int = None,
                  evalmax: int = None,
                  timemax: float = None,
                  session: dict = None,
                  infile: str = None,
                  outfile: str = None,
                  verbose: int = 0):
    return HillClimbing(function=function,
                        search_space=search_space,
                        initialize=initialize,
                        seed=seed,
                        saving=saving).run(itermax=itermax,
                                           evalmax=evalmax,
                                           timemax=timemax,
                                           session=session,
                                           infile=infile,
                                           outfile=outfile,
                                           verbose=verbose)


class SimulatedAnnealing(RandomSearch):

    def __init__(self, function: Function,
                 search_space: SearchSpace | Iterable[Binary] | Binary | ndBinary,
                 cooling: Cooling = NoneCooling(),
                 initialize: ndBits | Bits | bool | Any = None,
                 seed: int = None,
                 saving: bool = True):
        self.cooling = cooling
        super().__init__(function, search_space, initialize, seed, saving)
        if isinstance(initialize, bool) and not initialize:
            raise ValueError(
                'The simulated annealing algorithm must be initialized')

    @property
    def history_item(self):
        return dict(iter=self.niter,
                    eval=self.neval,
                    time=time.time() - self.seconds,
                    T=self.cooling.tmax if self.niter == 0 else self.T,
                    fact=self.fmin if self.niter == 0 else self.fact,
                    fmin=self.fmin,
                    f=self.f,
                    x=self.x.bits)

    def begin(self):
        self.temperatureK = self.cooling.temperatureK
        self.xact = self.xmin
        self.fact = self.fmin

    def kernel(self):
        self.T = self.temperatureK(self.niter)
        self.eval(self.search_space.neighbor(self.xact))
        if self.f < self.fact:
            self.xact = self.x
            self.fact = self.f
        elif (self.T > 0 and
              self.search_space.random.rand() <
              math.exp(-(self.f - self.fact) / self.T)):
            self.xact = self.x
            self.fact = self.f
        self.save()


def simulated_annealing(function: Function,
                        search_space: SearchSpace | Iterable[Binary] | Binary | ndBinary,
                        cooling: Cooling = NoneCooling(),
                        initialize: ndBits | Bits | bool | Any = None,
                        seed: int = None,
                        saving: bool = True,
                        itermax: int = None,
                        evalmax: int = None,
                        timemax: float = None,
                        session: dict = None,
                        infile: str = None,
                        outfile: str = None,
                        verbose: int = 0):
    return SimulatedAnnealing(function=function,
                              search_space=search_space,
                              cooling=cooling,
                              initialize=initialize,
                              seed=seed,
                              saving=saving).run(itermax=itermax,
                                                 evalmax=evalmax,
                                                 timemax=timemax,
                                                 session=session,
                                                 infile=infile,
                                                 outfile=outfile,
                                                 verbose=verbose)
