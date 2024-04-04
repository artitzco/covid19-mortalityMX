import numpy as np


def roulette(data, weight=1, size=1, k=1, replace=True, random=None):
    random = np.random if random is None else random
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
        if k == 1:
            return replace_samp(data, weight, size)
        sample = replaceK_samp(data, weight, size, k)
    else:
        if k == 1:
            sample = nonreplace_samp(data, weight, size, replace_samp)
        else:
            sample = nonreplace_samp(data, weight, size,
                                     lambda *x: replaceK_samp(*x, k=k))
    if len(sample) == size:
        return sample
    return sample[:size]


def tournament(data1, weight1, data2=None, weight2=None,
               size=None, replace=True, dim=2, random=None):
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
    else:
        dim = int(np.ceil(dim / 2))
        size = min(len(data1), len(data2)) // dim if size is None else size
        result = np.zeros(size, dtype=data1.dtype)
        index1 = random.choice(np.arange(len(data1)),
                               size=size * dim, replace=replace)
        index2 = random.choice(np.arange(len(data2)),
                               size=size * dim, replace=replace)

        for i in range(size):
            ind = np.arange(dim * i, dim * (i + 1))
            weightmax = -float('Inf')
            for data, weight, index in ((data1, weight1, index1[ind]),
                                        (data2, weight2, index2[ind])):
                for j in index:
                    if weight[j] > weightmax:
                        result[i] = data[j]
                        weightmax = weight[j]
    return result


def recombination(search_space, population, size,
                  crossing_probability=0.5, mutation_probability=None, mutation_scale=1):
    mutation_probability = ((mutation_scale / (search_space.binary.dimension * size))
                            if mutation_probability is None else mutation_probability)
    random = search_space.random
    newpop = np.zeros(size, dtype=population.x.dtype)
    for i in range(size):
        newpop[i] = search_space.mutation(
            search_space.crossing(*random.choice(population.x, size=2, replace=False),
                                  probability=crossing_probability), probability=mutation_probability)
    return newpop
