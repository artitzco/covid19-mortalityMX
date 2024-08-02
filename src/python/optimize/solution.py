from bitarray.util import ba2int, int2ba
from bitarray import bitarray
from collections import Counter
import numpy as np
import math


def entropy(data, universe, normalize=True):
    frequency = Counter(data)
    probabilities = {symbol: frequency[symbol] / len(data)
                     for symbol in frequency}
    entropy = -sum(p * math.log2(p) for p in probabilities.values() if p > 0)
    if normalize:
        return entropy / math.log2(len(universe))
    return entropy


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


class Solution:

    def __init__(self, random=None, seed=None, name=None, generator=None) -> None:
        self.random = random
        if seed is not None:
            self.seed = seed
        self.name = name
        if generator is None:
            self.generator = lambda _: None
        else:
            self.generator = generator

    @property
    def seed(self):
        raise AttributeError('The "seed" value is not available')

    @seed.setter
    def seed(self, seed):
        self.random.seed(seed)

    @property
    def random(self):
        return self._random

    @random.setter
    def random(self, random):
        if random is None:
            self._random = np.random
        else:
            self._random = random

    def new(self):
        return self.generator(self.random)

    def dimension(self, _): return 0

    def copy(self, x): return x

    def encode(self, x): return x

    def decode(self, x, to_dict=False):
        if to_dict:
            return {self.name: x}
        return x

    def write(self, x): return x

    def read(self, x): return x

    def mutate(self, x, prob=None): return x

    def cross(self, x, y, prob=0.5): return x

    def neighbor(self, x, epsilon=1): return x

    def entropy(self, x):
        return float('nan')

    def __repr__(self) -> str:
        return 'Solution()'

    def __str__(self) -> str:
        return 'Solution()'


class ndSolution(Solution, tuple):

    def __new__(cls, *args, random=None, seed=None, name=None, dimension=None, **kwargs):
        if len(kwargs) == 0:
            if dimension is not None:
                args = list(args) * dimension
        else:
            if len(args) > 0:
                raise ValueError(
                    '"args" values an "kwargs" values are not compatibles.')
            for name, sol in kwargs.items():
                sol.name = name
            args = kwargs.values()
        return super(ndSolution, cls).__new__(cls, args)

    def __init__(self, *_, random=None, seed=None, name=None, dimension=None, **kwargs) -> None:
        def generator(_): return [sol.new() for sol in self]
        super().__init__(random, seed, name, generator)
        for sol in self:
            sol.random = self.random
        self.dictable = len(kwargs) > 0

    def dimension(self, x):
        return sum([sol.dimension(x) for sol, x in zip(self, x)])

    def copy(self, x):
        return [sol.copy(x) for sol, x in zip(self, x)]

    def encode(self, x):
        return [sol.encode(x) for sol, x in zip(self, x)]

    def decode(self, x, to_dict=False):
        if to_dict:
            if self.dictable:
                dic = dict()
                for sol, x in zip(self, x):
                    dic.update(sol.decode(x, True))
                return dic
            else:
                return {self.name: [sol.decode(x) for sol, x in zip(self, x)]}
        return [sol.decode(x) for sol, x in zip(self, x)]

    def write(self, x):
        return [sol.write(x) for sol, x in zip(self, x)]

    def read(self, x):
        return [sol.read(x) for sol, x in zip(self, x)]

    def mutate(self, x, prob=None):
        return [sol.mutate(x, prob=prob) for sol, x in zip(self, x)]

    def cross(self, x, y, prob=0.5):
        return [sol.cross(x, y, prob=prob) for sol, x, y in zip(self, x, y)]

    def neighbor(self, x, epsilon=1):
        x = self.copy(x)
        if len(x) > 0:
            i = self.random.choice([i for i in range(len(self))],
                                   p=weightprob(
                                       [sol.dimension(x)
                                        for sol, x in zip(self, x)],
                abstolute=False),
                size=1)[0]
            x[i] = self[i].neighbor(x[i], epsilon=epsilon)
        return x

    def entropy(self, x):
        if len(x) > 0 and len(self) > 0:
            suma = 0
            dim = 0
            for j, sol in enumerate(self):
                dimx = np.mean([sol.dimension(x[i][j])
                                for i in range(len(x))])
                dim += dimx
                suma += dimx * sol.entropy([x[i][j]
                                            for i in range(len(x))])
            if dim > 0:
                return suma / dim
        return float('nan')

    def __repr__(self) -> str:
        names = ['' if sol.name is None else (sol.name+'=') for sol in self]
        text = ', '.join(
            ([name + sol.__repr__() for name, sol in zip(names, self)]))
        if len(self) == 1:
            return f'({text},)'
        return f'({text})'

    def __str__(self) -> str:
        names = ['' if sol.name is None else (sol.name+'=') for sol in self]
        text = ', '.join(
            ([name + sol.__str__() for name, sol in zip(names, self)]))
        if len(self) == 1:
            return f'({text},)'
        return f'({text})'


class Category(Solution):

    def __init__(self, items, random=None, seed=None, name=None) -> None:

        def generator(r): return self.keys[r.randint(len(self.keys))]

        super().__init__(random, seed, name, generator)
        if len(items) == 0:
            raise ValueError('The "items" length must be positive.')
        if isinstance(items, set):
            items = {item: item for item in items}
        elif not isinstance(items, dict):
            raise ValueError('The "items" instance must be a "dict" or "set".')
        self.keys = tuple(sorted(items.keys()))
        self.items = items

    def dimension(self, _):
        return 0 if len(self.keys) == 1 else 1

    def copy(self, x):
        return x

    def encode(self, x):
        for k, v in self.items.items():
            if x == v:
                return k
        raise ValueError('The "x" element is not in self items values.')

    def decode(self, x, to_dict=False):
        if to_dict:
            return {self.name: self.items[x]}
        return self.items[x]

    def write(self, x):
        return x

    def read(self, x):
        return x

    def mutate(self, x, prob=None):
        if len(self.keys) > 1:
            if prob is None:
                prob = 1
            if self.random.rand() < prob:
                return np.random.choice([k for k in self.keys if k != x])
        return x

    def cross(self, x, y, prob=0.5):
        return x if self.random.rand() < prob else y

    def neighbor(self, x, epsilon=1):
        if len(self.keys) > 1 and epsilon > 0:
            return np.random.choice([k for k in self.keys if k != x])
        return x

    def entropy(self, x):
        if len(self.keys) > 1:
            return entropy(x, self.keys)
        return 0

    def __repr__(self) -> str:
        cad = ', '.join([str(x) for x in self.keys])
        return f'Category({{{cad}}})'

    def __str__(self) -> str:
        cad = ', '.join([str(x) for x in self.keys])
        return f'{{{cad}}}'


class ndCategory(Category):

    def __init__(self, items, random=None, seed=None, name=None, solname=None, **solutions) -> None:
        super().__init__(items, random, seed, name)

        def generator(r):
            return [self.keys[r.randint(len(self.keys))],
                    * [sol.generator(r) for sol in self.solutions.values()]]
        self.generator = generator
        self.solname = solname
        for sol in solutions.values():
            sol.random = self.random
        self.solutions = {k: solutions[k]
                          for k in self.keys if k in solutions}

    def dimension(self, x):
        for i, (k, sol) in enumerate(self.solutions.items()):
            if k == x[0]:
                return sol.dimension(x[i+1])
        return super().dimension(x)

    def copy(self, x):
        return [x[0], *[sol.copy(x[i+1]) for i, sol in enumerate(self.solutions.values())]]

    def encode(self, x):
        return [x[0], *[sol.encode(x[i+1])
                        for i, sol in enumerate(self.solutions.values())]]

    def decode(self, x, to_dict=False):
        if to_dict:
            if self.solname is None:
                dic = {self.name: self.items[x[0]]}
                for i, sol in enumerate(self.solutions.values()):
                    dic.update(sol.decode(x[i+1], True))
                return dic
            else:
                solist = [sol.decode(x[i+1])
                          for i, sol in enumerate(self.solutions.values())]
                return {self.name: self.items[x[0]],
                        self.solname: solist[0] if len(solist) == 1 else solist}

        return [self.items[x[0]], *[sol.decode(x[i+1])
                                    for i, sol in enumerate(self.solutions.values())]]

    def write(self, x):
        return [x[0], *[sol.write(x[i+1]) for i, sol in enumerate(self.solutions.values())]]

    def read(self, x):
        return [x[0], *[sol.read(x[i+1]) for i, sol in enumerate(self.solutions.values())]]

    def mutate(self, x, prob=None):
        has_sol = False
        for i, (k, sol) in enumerate(self.solutions.items()):
            if x[0] == k:
                has_sol = True
                break
        if has_sol:
            di = sol.dimension(x[i+1])
            if di > 0:
                #### Ajuste de probabilidad ####
                if di > 3:
                    pi = 1 / (di - 3)
                else:
                    pi = 1
                #### Fin de ajuste ####
                if prob is None:
                    prob = di**-1
                swich_prob = (len(self.keys) - 1) / (len(self.keys) + di - 1)
                gate_prob = 1 - (1 - prob) ** di
                gate_prob /= 1 - (1 - swich_prob) * (1 - pi) ** di
                if self.random.rand() < gate_prob:
                    if self.random.rand() < swich_prob:
                        return [super().mutate(x[0], prob=di * pi),
                                *[sol.copy(x[i+1]) for i, sol in enumerate(self.solutions.values())]]
                    else:
                        return [x[0],
                                *[(sol.mutate(x[i+1],  prob=pi)
                                   if i == j else sol.copy(x[j+1]))
                                for j, sol in enumerate(self.solutions.values())]]
        return [super().mutate(x[0], prob=prob),
                *[sol.copy(x[i+1]) for i, sol in enumerate(self.solutions.values())]]

    def cross(self, x, y, prob=0.5):
        return [super().cross(x[0], y[0], prob=prob),
                *[sol.cross(x[i+1], y[i+1], prob=prob) for i, sol in enumerate(self.solutions.values())]]

    def neighbor(self, x, epsilon=1):
        has_sol = False
        for i, (k, sol) in enumerate(self.solutions.items()):
            if x[0] == k:
                has_sol = True
                break
        if has_sol:
            di = sol.dimension(x[i+1])
            swich_prob = (len(self.keys) - 1) / (len(self.keys) + di - 1)
            if self.random.rand() > swich_prob:
                return [x[0],
                        *[(sol.neighbor(x[i+1],  epsilon=epsilon)
                            if i == j else sol.copy(x[j+1]))
                            for j, sol in enumerate(self.solutions.values())]]
        return [super().neighbor(x[0], epsilon=epsilon),
                *[sol.copy(x[i+1]) for i, sol in enumerate(self.solutions.values())]]

    def entropy(self, x):
        if len(x) > 0:
            if len(self.solutions) > 0:
                dim = super().dimension(False)
                suma = dim * super().entropy([x[0] for x in x])
                for j, sol in enumerate(self.solutions.values()):
                    dimx = np.mean([sol.dimension(x[i][j+1])
                                    for i in range(len(x))])
                    dim += dimx
                    suma += dimx * sol.entropy([x[i][j+1]
                                                for i in range(len(x))])
                return suma / dim
            return super().entropy([x[0] for x in x])
        return float('nan')

    def __repr__(self) -> str:
        cad = ', '.join([f'{k}={sol.__repr__()}'
                        for k, sol in self.solutions.items()])
        return super().__repr__()[:-1] + ', '+cad + ')'

    def __str__(self) -> str:
        cad = ' + '.join([sol.__repr__() for sol in self.solutions.values()])
        return super().__str__() + ' | ' + cad


def Bits(x):
    return bitarray(x, endian='big')


def adjus_bits(bits, nbits):
    if len(bits) > nbits:
        return Bits(bits[-nbits:])
    if len(bits) < nbits:
        return Bits(nbits - len(bits)) + bits
    return bits


class Binary(Solution):

    def __init__(self, nbits, random=None, seed=None, name=None) -> None:
        self.nbits = nbits
        self.size = 2 ** self.nbits

        def generator(r):
            return Bits(list(r.choice([0, 1],
                                      size=self.nbits)))
        super().__init__(random, seed, name, generator)

    def dimension(self, _):
        return self.nbits

    def copy(self, x):
        return Bits(x)

    def encode(self, x):
        if self.nbits == 0:
            return Bits('')
        return adjus_bits(
            int2ba(x, endian='big'), self.nbits)

    def decode(self, x, to_dict=False):
        if to_dict:
            return {self.name: ba2int(x)}
        return ba2int(x)

    def write(self, x):
        return ''.join([str(x) for x in x])

    def read(self, x):
        if self.nbits == 0:
            return Bits('')
        bits = Bits(x) if isinstance(x, str) else int2ba(x, endian='big')
        return adjus_bits(bits, self.nbits)

    def mutate(self, x, prob=None):
        bits = Bits(x)
        if self.nbits > 0:
            if prob is None:
                prob = 1 / self.nbits
            count = self.random.binomial(len(x), prob)
            if count > 0:
                index = self.random.choice(len(x), count, replace=False)
                for i in index:
                    bits[i] = not bits[i]
        return bits

    def cross(self, x, y,  prob=0.5):
        bits = Bits(x)
        if self.nbits > 0:
            count = self.random.binomial(len(x), prob)
            if count > 0:
                index = self.random.choice(len(x), count, replace=False)
                for i in index:
                    bits[i] = y[i]
        return bits

    def neighbor(self, x, epsilon=1):
        bits = Bits(x)
        if self.nbits > 0:
            count = self.random.randint(1, 1 + epsilon)
            index = self.random.choice(len(x), count, replace=False)
            for i in index:
                bits[i] = not bits[i]
        return bits

    def entropy(self, x):
        if self.nbits > 0 and len(x) > 0:
            return sum([entropy([x[i][j]
                                 for i in range(len(x))], {0, 1})
                        for j in range(self.nbits)]) / self.nbits
        return 0

    def __repr__(self):
        return f'Binary({self.nbits})'

    def __str__(self):
        return f'{{0,...,{self.size-1}}}n={self.nbits}'


class floatBinary(Binary):

    def __init__(self, a, b, nbits=None, digits=None, dmin=None, random=None, seed=None, name=None) -> None:
        a, b = float(a), float(b)
        if a > b:
            raise ValueError('The "a" value cannot be greater than "b" value')
        if a < b:
            if nbits is None:
                if digits is not None:
                    nbits = np.log2(1 + (b - a) * np.exp(digits * np.log(10)))
                elif dmin is not None:
                    if dmin <= 0:
                        raise ValueError('The "dmin" value must be positive')
                    nbits = np.log2(1 + (b - a) / dmin)
            nbits = 1 if nbits is None else nbits
            nbits = int(max(1, np.ceil(nbits)))
            super().__init__(nbits, random, seed, name)
            self.digits = int(-np.log10((b - a) / (self.size - 1)))
            self.dmin = (b - a) / (self.size - 1)
        else:
            super().__init__(0, random, seed, name)
            self.digits, self.dmin = 0, 0

        self.a, self.b = a, b

    def dimension(self, _):
        if self.a < self.b:
            return self.nbits
        return 0

    def encode(self, x):
        if self.a == self.b:
            return Bits(0)
        return super().encode(int((self.size - 1) * (x - self.a) / (self.b - self.a)))

    def decode(self, x, to_dict=False):
        if self.a == self.b:
            x = self.a
        else:
            x = self.a + (self.b - self.a) * super().decode(x) / (self.size-1)
        if to_dict:
            return {self.name: x}
        return x

    def __repr__(self):
        return f'floatBinary({self.a}, {self.b}, {self.nbits})'

    def __str__(self):
        if self.a == self.b:
            return f'[{self.a}]n={self.nbits}'
        return f'[{self.a}, {self.b}]n={self.nbits}'


class intBinary(Binary):

    def __init__(self, a, b, nbits=None, digits=None, dmin=None, random=None, seed=None, name=None) -> None:
        a, b = int(a), int(b)
        if a > b:
            raise ValueError('The "a" value cannot be greater than "b" value')
        if a < b:
            if nbits is None:
                if digits is not None:
                    if digits > 0:
                        raise ValueError(
                            'The "digits" value cannot be positive')
                    nbits = np.log2(1 + (b - a) * np.exp(digits * np.log(10)))
                elif dmin is not None:
                    if dmin < 1:
                        raise ValueError(
                            'The "dmin" value cannot be less than 1')
                    nbits = np.log2(1 + (b - a) / dmin)
            nbits = 1 if nbits is None else nbits
            nbits = int(min(max(1, np.ceil(nbits)),
                        np.ceil(np.log2(1 + (b - a)))))
            super().__init__(nbits, random, seed, name)
            self.digits = int(-np.log10((b - a) / (self.size - 1)))
            self.dmin = (b - a) / (self.size - 1)
        else:
            super().__init__(0, random, seed, name)
            self.digits, self.dmin = 0, 0

        self.a, self.b = a, b

    def dimension(self, _):
        if self.a < self.b:
            return self.nbits
        return 0

    def encode(self, x):
        if self.a == self.b:
            return Bits(0)
        return super().encode(int((self.size - 1) * (x - self.a) / (self.b - self.a)))

    def decode(self, x, to_dict=False):
        if self.a == self.b:
            x = self.a
        else:
            x = round(self.a + (self.b - self.a) *
                      super().decode(x) / (self.size-1))
        if to_dict:
            return {self.name: x}
        return x

    def __repr__(self):
        return f'intBinary({self.a}, {self.b}, {self.nbits})'

    def __str__(self):
        if self.a == self.b:
            return f'{{{self.a}}}n={self.nbits}'
        if self.a == self.b - 1:
            return f'{{{self.a}, {self.b}}}n={self.nbits}'
        return f'{{{self.a},...,{self.b}}}n={self.nbits}'
