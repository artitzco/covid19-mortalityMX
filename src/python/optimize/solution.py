from bitarray.util import ba2int, int2ba
from bitarray import bitarray
from typing import Callable
import numpy as np
import inspect
import sys


class Solution:
    def __init__(self, random=None, seed=None, generator=None, name=None) -> None:
        self.random = random
        if seed is not None:
            self.seed = seed
        self.generator = generator
        self.name = name
        self.aggregates = dict()

    def agg(self, name, fun=None, **kwargs):
        fun = name if fun is None else fun
        funargs = {k: v for k, v in kwargs.items() if isinstance(v, Callable)}
        kwargs = {k: v for k, v in kwargs.items() if k not in funargs}
        if isinstance(fun, Callable):
            self.aggregates[name] = fun, funargs, kwargs
        elif isinstance(fun, str):
            self.aggregates[name] = getattr(self, fun), funargs, kwargs
        return self

    def do(self, name, *args, **kwargs):
        if name in self.aggregates:
            fun, funargs, _ = self.aggregates[name]
            fargs = {n: f(**{k: v for k, v in kwargs.items()
                             if k in inspect.signature(f).parameters.keys()})
                     for n, f in funargs.items()}
            return fun(*args, **fargs, **_)
        if hasattr(self, name):
            return getattr(self, name)(*args, **kwargs)
        raise AttributeError(f'Function "{name}" not found')

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

    @property
    def generator(self):
        if self._generator is None:
            raise ValueError('No solution generator has been added')
        return self._generator

    @generator.setter
    def generator(self, generator):
        self._add_generator(generator)

    def _add_generator(self, generator):
        self._generator = generator

    def is_in_range(self, _):
        return True

    def new(self):
        x = self.generator(self.random)
        while not self.is_in_range(x):
            x = self.generator(self.random)
        return x

    def distance(self, x, y, d=1): return 0

    def dimension(self, _): return 0

    def encode(self, x): return x

    def decode(self, x, to_dict=False):
        if to_dict:
            return {self.name: x}
        return x

    def write(self, x): return x

    def read(self, x): return x

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
        super().__init__(random, seed, None, name)
        for sol in self:
            sol.random = self.random

    def agg(self, name, fun=None, **kwargs):
        for sol in self:
            sol.agg(name, fun, **kwargs)
        return self

    def do(self, name, *args, **kwargs):
        args = [x if isinstance(x, list) else [x]*len(self) for x in args]
        return [sol.do(name, *arg, **kwargs) for sol, arg in zip(self, zip(*args))]

    def _add_generator(self, _):
        self._generator = lambda _: [sol.new() for sol in self]

    def distance(self, x, y, d=1):
        if d == 0:
            return np.mean([sol.distance(x, y, d) for sol, x, y in zip(self, x, y)])
        if d < float('inf'):
            return sum([sol.distance(x, y, d) for sol, x, y in zip(self, x, y)])**(1/d)
        return np.max([sol.distance(x, y, d) for sol, x, y in zip(self, x, y)])

    def dimension(self, x):
        return sum([sol.dimension(x) for sol, x in zip(self, x)])

    def encode(self, x):
        return [sol.encode(x) for sol, x in zip(self, x)]

    def decode(self, x, to_dict=False):
        if to_dict:
            dic = dict()
            for sol, x in zip(self, x):
                dic.update(sol.decode(x, True))
            return dic
        return [sol.decode(x) for sol, x in zip(self, x)]

    def write(self, x):
        return [sol.write(x) for sol, x in zip(self, x)]

    def read(self, x):
        return [sol.read(x) for sol, x in zip(self, x)]

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


class Float(Solution):
    def __init__(self, a='-inf', b='inf', decimals=None, random=None, seed=None, generator=None, name=None) -> None:
        self.a = float(a)
        self.b = float(b)
        self.decimals = decimals
        if self.b < self.a:
            raise ValueError(
                'The "b" value cannot be less than the "a" value.')
        super().__init__(random, seed, generator, name)

    def Uniform(a, b):
        return lambda r:  r.uniform(a, b)

    def Normal(mu, sigma):
        return lambda r:  mu + sigma * r.standard_normal()

    def _add_generator(self, generator):
        a, b, max = self.a, self.b, float('inf')
        if generator is None:
            generator = 'uniform' if a > -max and b < max else 'normal'
        if a == b:
            self._generator = lambda _: a
        elif generator == 'uniform':
            if a > -max and b < max:
                self._generator = Float.Uniform(a, b)
            else:
                raise ValueError(
                    'The uniform generator is only available when "a" and "b" are bounded.')
        elif generator == 'normal':
            if a > -max and b < max:
                self._generator = Float.Normal(
                    (a + b) / 2.0, (b - a) / 4.0)
            elif a > -max:
                self._generator = Float.Normal(a, 1)
            elif b < max:
                self._generator = Float.Normal(b, 1)
            else:
                self._generator = Float.Normal(0, 1)
        else:
            self._generator = generator

    def is_in_range(self, x):
        return self.a <= x <= self.b

    def distance(self, x, y, d=1):
        if d > 0 and d < float('inf'):
            return abs(x - y) ** d
        return abs(x - y)

    def dimension(self, _):
        return 0 if self.a == self.b else 1

    def encode(self, x):
        return float(self.write(x))

    def decode(self, x, to_dict=False):
        if to_dict:
            return super().decode(float(self.write(x)), True)
        return float(self.write(x))

    def write(self, x):
        if self.decimals is None:
            return str(x)
        return f'{{:.{self.decimals}f}}'.format(x)

    def read(self, x):
        x = float(x)
        if not self.is_in_range(x):
            raise ValueError('The number "x" exceeds the bounds')
        return x

    def mutate(self, x, sigma=1):
        if self.dimension(x) == 0:
            return self.a
        y = self.encode(x + sigma * self.random.standard_normal())
        while not self.is_in_range(y):
            y = self.encode(x + sigma * self.random.standard_normal())
        return y

    def cross(self, x, y, scale=0.5):
        if self.dimension(x) == 0:
            return self.a
        z = self.encode(scale * x + (1 - scale) * y)
        if not self.is_in_range(z):
            raise ValueError('The lineal crossing is out of range.')
        return z

    def __repr__(self) -> str:
        if self.a == self.b:
            return f'Float({self.a})'
        return f'Float({self.a}, {self.b})'

    def __str__(self) -> str:
        if self.a == self.b:
            return f'[{self.a}]'
        return f'[{self.a}, {self.b}]'


class Int(Float):
    def __init__(self, a=-sys.maxsize, b=sys.maxsize, random=None, seed=None, generator=None, name=None) -> None:
        super().__init__(a, b, None, random, seed, generator, name)
        self.a = int(a)
        self.b = int(b)

    def Uniform(a, b):
        return lambda r:  r.randint(a, b + 1)

    def Normal(mu, sigma):
        return lambda r:  round(mu + sigma * r.standard_normal())

    def _add_generator(self, generator):
        a, b, max = self.a, self.b, sys.maxsize
        if generator is None:
            generator = 'uniform' if a > -max and b < max else 'normal'
        if a == b:
            self._generator = lambda _: a
        elif generator == 'uniform':
            if a > -max and b < max:
                self._generator = Int.Uniform(a, b)
            else:
                raise ValueError(
                    'The uniform generator is only available when "a" and "b" are bounded.')
        elif generator == 'normal':
            if a > -max and b < max:
                self._generator = Int.Normal(
                    (a + b) / 2.0, (b - a) / 4.0)
            elif a > -max:
                self._generator = Int.Normal(a, 1)
            elif b < max:
                self._generator = Int.Normal(b, 1)
            else:
                self._generator = Int.Normal(0, 1)
        else:
            self._generator = generator

    def encode(self, x):
        return int(self.write(x))

    def decode(self, x, to_dict=False):
        if to_dict:
            return super().decode(int(self.write(x)), True)
        return int(self.write(x))

    def write(self, x):
        return str(self.read(x))

    def read(self, x):
        x = int(x)
        if not self.is_in_range(x):
            raise ValueError('The number "x" exceeds the bounds')
        return x

    def mutate(self, x, sigma=1):
        if self.dimension(x) == 0:
            return self.a
        y = round(x + sigma * self.random.standard_normal())
        while not self.is_in_range(y):
            y = round(x + sigma * self.random.standard_normal())
        return y

    def cross(self, x, y, scale=0.5):
        if self.dimension(x) == 0:
            return self.a
        y = round(scale * x + (1 - scale) * y)
        if not self.is_in_range(y):
            raise ValueError('The lineal crossing is out of range.')
        return y

    def __repr__(self) -> str:
        if self.a == self.b:
            return f'Int({self.a})'
        return f'Int({self.a}, {self.b})'

    def __str__(self) -> str:
        if self.a == self.b:
            return f'{{{self.a}}}'
        return f'{{{self.a},...,{self.b}}}'


class Category(Solution):
    def __init__(self, items, random=None, seed=None, generator=None, name=None) -> None:

        super().__init__(random, seed, generator, name)
        if len(items) == 0:
            raise ValueError('The "items" length must be positive.')
        if isinstance(items, set):
            items = {item: item for item in items}
        elif not isinstance(items, dict):
            raise ValueError('The "items" instance must be a "dict" or "set".')
        self.keys = tuple(sorted(items.keys()))
        self.items = items

    def _add_generator(self, generator):
        if generator is None:
            self._generator = lambda r: self.keys[r.randint(len(self.keys))]
        else:
            self._generator = generator

    def is_in_range(self, x):
        return x in self.items

    def dimension(self, _):
        return 0 if len(self.keys) == 1 else 1

    def distance(self, x, y, _):
        if x == y:
            return 0
        return 1

    def encode(self, x):
        for k, v in self.items.items():
            if x == v:
                return k
        raise ValueError('The "x" element is not in self items values.')

    def decode(self, x, to_dict=False):
        if not self.is_in_range(x):
            raise ValueError('The "x" element is not in self items.')
        if to_dict:
            return super().decode(self.items[x], True)
        return self.items[x]

    def write(self, x):
        if not self.is_in_range(x):
            raise ValueError('The "x" element is not in self items.')
        return x

    def read(self, x):
        if not self.is_in_range(x):
            raise ValueError('The "x" element is not in self items.')
        return x

    def mutate(self, x, prob=None, scale=1):
        prob = scale if prob is None else scale * prob
        if self.random.rand() < prob:
            return np.random.choice([k for k in self.keys if k != x])
        return x

    def cross(self, x, y, prob=0.5):
        x, y = self.read(x), self.read(y)
        if self.dimension(x) == 0:
            return self.keys[0]
        return x if self.random.rand() < prob else y

    def __repr__(self) -> str:
        cad = ', '.join([str(x) for x in self.keys])
        return f'Category({{{cad}}})'

    def __str__(self) -> str:
        cad = ', '.join([str(x) for x in self.keys])
        return f'{{{cad}}}'


class ndCategory(Category):

    def __init__(self, items, random=None, seed=None, generator=None, name=None, solname=None, delta=1, **solutions) -> None:
        super().__init__(items, random, seed, generator, name)
        self.solname = solname
        for sol in solutions.values():
            sol.random = self.random
        self.solutions = {k: solutions[k]
                          for k in self.keys if k in solutions}
        if isinstance(delta, (int, float)):
            self.delta = [delta] * len(self.solutions)
        else:
            self.delta = delta

    def _add_generator(self, generator):
        if generator is None:
            def generator(r):
                return [self.keys[r.randint(len(self.keys))],
                        * [sol.generator(r) for sol in self.solutions.values()]]
        self._generator = generator

    def is_in_range(self, x):
        for y, sol in zip(x[1:], self.solutions.values()):
            if not sol.is_in_range(y):
                return False
        return x[0] in self.items

    def distance(self, x, y, d=1):
        for i, (k, sol) in enumerate(self.solutions.items()):
            if k == x[0] and k == y[0]:
                return sol.distance(x[i+1], y[i+1], d)
        return super().distance(x[0], y[0], d)

    def dimension(self, x):
        for i, (k, sol) in enumerate(self.solutions.items()):
            if k == x[0]:
                return sol.dimension(x[i+1])
        return super().dimension(x)

    def encode(self, x):
        if not super().is_in_range(x[0]):
            raise ValueError('The "x[0]" element is not in self items.')
        return [x[0], *[sol.encode(x[i+1])
                        for i, sol in enumerate(self.solutions.values())]]

    def decode(self, x, to_dict=False):
        if not super().is_in_range(x[0]):
            raise ValueError('The "x[0]" element is not in self items.')
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
        if not super().is_in_range(x[0]):
            raise ValueError('The "x[0]" element is not in self items.')
        return [x[0], *[sol.write(x[i+1]) for i, sol in enumerate(self.solutions.values())]]

    def read(self, x):
        if not super().is_in_range(x[0]):
            raise ValueError('The "x[0]" element is not in self items.')
        return [x[0], *[sol.read(x[i+1]) for i, sol in enumerate(self.solutions.values())]]

    def mutate(self, x, prob=None, scale=1):
        has_sol = False
        for i, (k, sol) in enumerate(self.solutions.items()):
            if x[0] == k:
                has_sol = True
                break
        if has_sol:
            di = sol.dimension(x[i+1])
            if prob is None:
                prob = di**-1
            prob = scale * prob
            probs = (len(self.keys) - 1) / (len(self.keys) + di - 1)
            probi = self.delta[i] * scale * di**-1
            probc = di * probi / self.delta[i]
            prob0 = (1 - (1-prob)**di) / \
                (1 - probs * (1-probc) - (1-probs) * (1-probi)**di)
            if self.random.rand() < prob0:
                if self.random.rand() < probs:
                    return [super().mutate(x[0], prob=probc, scale=1), *x[1:]]
                else:
                    return [*x[:i+1], sol.do('mutate', x[i+1],  prob=probi, scale=1), *x[i+2:]]
            return x.copy()
        else:
            return [super().mutate(x[0], prob=prob, scale=scale), *x[1:]]

    def __repr__(self) -> str:
        cad = ', '.join([f'{k}={sol.__repr__()}'
                        for k, sol in self.solutions.items()])
        return super().__repr__()[:-1] + ', '+cad + ')'

    def __str__(self) -> str:
        cad = ' + '.join([sol.__repr__() for sol in self.solutions.values()])
        return super().__str__() + ' | ' + cad


class Bits(bitarray):

    def __new__(cls, _):
        return super().__new__(cls, _, endian='big')

    def __init__(self, _):
        super().__init__()

    def mutate(self, prob, random):
        bits = Bits(self)
        if len(self) > 0:
            count = random.binomial(len(self), prob)
            if count > 0:
                index = random.choice(len(self), count, replace=False)
                for i in index:
                    bits[i] = not bits[i]
        return bits

    def cross(self, inputbits,  prob, random):
        if len(inputbits) != len(self):
            raise ValueError(
                'self and inputbits cannot have different lentgths')
        bits = Bits(self)
        if len(self) > 0:
            count = random.binomial(len(self), prob)
            if count > 0:
                index = random.choice(len(self), count, replace=False)
                for i in index:
                    bits[i] = inputbits[i]
        return bits

    def neighbor(self, epsilon, random):
        bits = Bits(self)
        if epsilon > 0:
            count = random.randint(1, epsilon + 1)
            index = random.choice(len(self), count, replace=False)
            for i in index:
                bits[i] = not bits[i]
        return bits

    def neighborhood(self):
        def seti(i):
            bits = Bits(self)
            bits[i] = not bits[i]
            return bits
        return [seti(i) for i in range(len(self))]

    def __repr__(self):
        return f'Bits({str(self)})'

    def __str__(self):
        return ''.join([str(bit) for bit in self])


class Binary(Solution):

    def __init__(self, nbits, random=None, seed=None, generator=None, name=None) -> None:
        self.nbits = nbits
        self.size = 2 ** self.nbits
        super().__init__(random, seed, generator, name)

    def _add_generator(self, generator):
        if generator is None:
            self.generator = lambda r:  \
                Bits(list(r.choice([0, 1], size=self.nbits)))
        else:
            self._generator = generator

    def is_in_range(self, _):
        return True

    def distance(self, x, y, d=1):
        if d == 0:
            return np.mean([abs(x-y) for x, y in zip(x, y)])
        if d == 1:
            return sum([abs(x-y) for x, y in zip(x, y)])
        if d == float('inf'):
            return max([abs(x-y) for x, y in zip(x, y)])
        return sum([abs(x-y)**d for x, y in zip(x, y)])**(1/d)

    def dimension(self, _):
        return self.nbits

    def encode(self, x):
        bits = int2ba(x, endian='big')
        n = len(bits)
        if n > self.nbits:
            return Bits(bits[-self.nbits:])
        if n < self.nbits:
            return Bits(bitarray(self.nbits - n) + bits)
        return Bits(bits)

    def decode(self, x, to_dict=False):
        if to_dict:
            return super().decode(ba2int(x), True)
        return ba2int(x)

    def write(self, x):
        return ''.join([str(x) for x in x])

    def read(self, x):
        if isinstance(x, str):
            return Bits(x)
        else:
            return self.encode(x)

    def mutate(self, x, prob=None, scale=1, factor=1):
        prob = scale / self.nbits if prob is None else scale * prob
        prob = 1 - (1 - prob) * factor**(1/self.nbits)
        return x.mutate(prob, self.random)

    def cross(self, x, y, prob=0.5):
        return x.cross(y, prob, self.random)

    def neighbor(self, x, epsilon=1):
        return x.neighbor(epsilon, self.random)

    def __repr__(self):
        return f'Binary({self.nbits})'

    def __str__(self):
        return f'{{0,...,{self.size-1}}}n={self.nbits}'


class floatBinary(Binary):

    def __init__(self, a, b, nbits=None, digits=None, dmin=None, random=None, seed=None, generator=None, name=None) -> None:
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
            super().__init__(nbits, random, seed, generator, name)
            self.digits = int(-np.log10((b - a) / (self.size - 1)))
            self.dmin = (b - a) / (self.size - 1)
        else:
            super().__init__(0, random, seed, generator, name)
            self.digits, self.dmin = 0, 0

        self.a, self.b = a, b

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

    def __init__(self, a, b, nbits=None, digits=None, dmin=None, random=None, seed=None, generator=None, name=None) -> None:
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
            super().__init__(nbits, random, seed, generator, name)
            self.digits = int(-np.log10((b - a) / (self.size - 1)))
            self.dmin = (b - a) / (self.size - 1)
        else:
            super().__init__(0, random, seed, generator, name)
            self.digits, self.dmin = 0, 0

        self.a, self.b = a, b

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


def entropy(data, nd=False):
    def log(x): return 0 if x == 0 else np.log(x)
    h = 0
    if nd:
        for j in range(len(data[0])):
            h += entropy([d[j] for d in data], nd=False)
        return h / len(data[0])

    for i in range(len(data[0])):
        p = sum([bits[i] for bits in data]) / len(data)
        h += -p * log(p) - (1-p) * log(1-p)

    return h / len(data[0])


def diversity(data, index, solution, d=float('inf')):
    axis = data[index]
    div = 0
    for x in data:
        div += solution.distance(axis, x, d)
    return div / (len(data) - 1)
