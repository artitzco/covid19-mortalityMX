from src.python.optimize.algorithms import RandomSearch, SearchSpace, Function
from src.python.optimize.binary import Binary, ndBinary
from typing import Iterable
import pandas as pd
import numpy as np
import time


def roulette(data, weight=1, size=1, k=1, replace=True, random=None):
    random = np.random if random is None else random
    df = None
    if isinstance(data, pd.DataFrame):
        df = data
        data = data.index

    weight = (np.full(len(data), weight)
              if isinstance(weight, (int, float)) else weight)

    def replace_samp(data, weight, size):
        prob = weight / weight.sum()
        cumprob = np.concatenate(([0], (prob / prob.sum()).cumsum()))
        pos = np.zeros(size, dtype=int)
        rand = random.rand(size)
        rand.sort()
        j = 1
        for i, r in enumerate(rand):
            for j in range(j, len(cumprob)):
                if r < cumprob[j]:
                    pos[i] = j - 1
                    break
        return data[pos]

    def replaceK_samp(data, weight, size, k):
        n = len(data)
        basis = replace_samp(np.arange(n), weight, int(np.ceil(size / k)))
        sample = [basis]
        for i in range(1, k):
            basisk = np.zeros(len(basis), dtype=int)
            for j in range(len(basisk)):
                basisk[j] = int((basis[j] + i * n / k) % n)
            sample.append(basisk)
        sample = np.concatenate(sample)
        return data[sample]

    def nonreplace_samp(data, weight, size, replace_samp):
        index = np.arange(len(data))
        locs = np.ones(len(data), dtype=bool)
        locsize = 0
        while locsize < size:
            locs[replace_samp(index[locs], weight[locs],
                              size - locsize)] = False
            locsize = len(data) - sum(locs)
        return data[~locs]
    if replace:
        sample = (replace_samp(data, weight, size) if k == 1
                  else replaceK_samp(data, weight, size, k))
    else:
        sample = (nonreplace_samp(data, weight, size, replace_samp) if k == 1
                  else nonreplace_samp(data, weight, size, lambda *x: replaceK_samp(*x, k=k)))
    if df is not None:
        df = df.loc[sample[:len(data)]]
        df.reset_index(inplace=True, drop=True)
        return df
    return sample[:len(data)]


def tournament(data1, weight1, data2=None, weight2=None,
               size=None, replace=True, dim=2, random=None):
    isdf = False
    if isinstance(data1, pd.DataFrame):
        isdf = True
        df1 = data1
        data1 = data1.index
        if data2 is not None:
            df2 = data2
            data2 = data2.index

    random = np.random if random is None else random
    if data2 is None:
        size = len(data1) // dim if size is None else size
        result = np.zeros(size, dtype=data1.dtype)
        index1 = random.choice(np.arange(len(data1)),
                               size=size * dim, replace=replace)
        for i in range(size):
            ind = np.arange(dim * i, dim * (i + 1))
            weightmax = -float('Inf')
            for j in index1[ind]:
                if weight1[j] > weightmax:
                    result[i] = data1[j]
                    weightmax = weight1[j]
        return df1.iloc[result] if isdf else result

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


def recombination(search_space, data, size,
                  crossing_probability=0.5, mutation_probability=None, mutation_scale=1):
    mutation_probability = ((mutation_scale / (search_space.binary.dimension * size))
                            if mutation_probability is None else mutation_probability)
    random = search_space.random
    newpop = np.zeros(size, dtype=data.dtype)
    for i in range(size):
        newpop[i] = search_space.mutation(
            search_space.crossing(*random.choice(data, size=2, replace=False),
                                  probability=crossing_probability), probability=mutation_probability)
    return newpop


class GeneticSearch(RandomSearch):

    def __init__(self, function: Function,
                 search_space: SearchSpace | Iterable[Binary] | Binary | ndBinary,
                 population: pd.DataFrame | dict,
                 seed: int = None):
        super().__init__(function, search_space, False, seed, True)
        self.population = pd.DataFrame(
            dict(f=population['f'], x=population['x']))

    @property
    def history_item(self):
        if self.kernel_startup:
            self.kernel_startup = False
            population = self.population
        else:
            population = None

        return dict(iter=self.niter,
                    eval=self.neval,
                    time=time.time() - self.seconds,
                    fmin=self.fmin,
                    f=self.f,
                    x=self.x.bits,
                    population=population)

    @property
    def session(self):
        return pd.Series(dict(iter=self.niter,
                              eval=self.neval,
                              time=time.time() - self.seconds,
                              fmin=self.fmin,
                              xmin=None if self.xmin is None else self.xmin.bits,
                              history=pd.DataFrame(self.history),
                              population=self.population))

    def begin(self):
        self.population['x'] = [
            self.search_space.new(x) for x in self.population.x]

    def kernel(self):
        self.kernel_startup = True
        parents = GeneticSearch.selection(
            self.search_space, self.population)
        x = GeneticSearch.reproduction(
            self.search_space, self.population, parents)
        f = np.zeros(len(x), dtype=x.dtype)
        for i in range(len(x)):
            f[i] = self.eval(x[i])
            self.save()
        self.population = GeneticSearch.replacement(
            self.search_space, self.population, pd.DataFrame(dict(f=f, x=x)))
        GeneticSearch.elitism(
            self.search_space, self.population, self.history, (self.fmin, self.xmin))

    def run(self, itermax: int = None, evalmax: int = None, timemax: float = None, session: dict = None, infile: str = None, outfile: str = None, verbose: int = 0):
        session = super().run(itermax, evalmax, timemax, session, infile, outfile, verbose)
        subhistory = session.history[~session.history.population.isna()]
        population = (subhistory.population.tolist() + [self.population])
        group = session.history.groupby('iter')
        data = pd.DataFrame(
            {name:
             ([0] + group[name].max().to_list()
              if name != 'fmin' else
              [pop.f.min() for pop in population])
             for name in ['iter', 'eval', 'time', 'fmin']})
        for i, pop in enumerate(population):
            population[i] = pd.concat(
                [pd.DataFrame(data.iloc[i].to_dict(), index=pop.index), pop], axis=1)
        population = pd.concat(population, axis=0)
        population['iter'] = pd.Series(population['iter'], dtype=int)
        population['eval'] = pd.Series(population['eval'], dtype=int)
        session['population'] = population
        return session

    def selection(search_space: SearchSpace, population: pd.DataFrame):
        return roulette(population,
                        weight=population.f.max() - population.f,
                        size=len(population) // 2,
                        replace=False,
                        random=search_space.random)

    def reproduction(search_space: SearchSpace, population: pd.DataFrame, parents: pd.DataFrame):
        return recombination(search_space, parents.x, size=len(population))

    def replacement(search_space: SearchSpace, population: pd.DataFrame, children: pd.DataFrame):
        return tournament(population, population.f.max()-population.f,
                          children, children.f.max()-children.f,
                          size=int(len(population)),
                          replace=False,
                          random=search_space.random)

    def elitism(search_space: SearchSpace, population: pd.DataFrame, history: pd.DataFrame, minimum: tuple):
        fmin, xmin = minimum
        fmax = float('-Inf')
        for i, f in enumerate(population.f):
            if f > fmax:
                fmax, imax = f, i
        population.loc[imax, 'f'] = fmin
        population.loc[imax, 'x'] = xmin


def genetic_search(function: Function,
                   search_space: SearchSpace | Iterable[Binary] | Binary | ndBinary,
                   population: pd.DataFrame | dict,
                   seed: int = None,
                   itermax: int = None,
                   evalmax: int = None,
                   timemax: float = None,
                   session: dict = None,
                   infile: str = None,
                   outfile: str = None,
                   verbose: int = 0):
    return GeneticSearch(function=function,
                         search_space=search_space,
                         population=population,
                         seed=seed).run(itermax=itermax,
                                        evalmax=evalmax,
                                        timemax=timemax,
                                        session=session,
                                        infile=infile,
                                        outfile=outfile,
                                        verbose=verbose)
