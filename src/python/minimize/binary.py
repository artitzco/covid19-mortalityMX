
from bitarray.util import ba2int, int2ba
from bitarray import bitarray
import numpy as np


def int_encoder(x, nbits):
    bits = int2ba(x, endian='big')
    n = len(bits)
    if n > nbits:
        return bits[-nbits:]
    if n < nbits:
        return bitarray(nbits - n) + bits
    return bits


def int_decoder(x):
    return ba2int(x)


def cat_encoder(x, keys):
    bits = bitarray(len(keys))
    for i, y in enumerate(keys):
        if x == y:
            bits[i] = 1
            return bits
    raise ValueError(f'"{x}" is not a value from "items"')


def cat_decoder(x, keys, items):
    for i, x in enumerate(x):
        if x == 1:
            return items[keys[i]]


class Bits(bitarray):

    def __new__(cls, _, decoder=None, random=None, name=None):
        return super().__new__(cls, _, endian='big')

    def __init__(self, _, decoder=None, random=None, name=None):
        super().__init__()
        self.decoder = (int_decoder
                        if decoder is None else decoder)
        self.random = (np.random.RandomState()
                       if random is None else random)
        self.name = name

    @property
    def value(self):
        return self.decoder(self)

    @property
    def bits(self):
        return ''.join([str(bit) for bit in self])

    @property
    def value_dict(self):
        return {self.name: self.value}

    @property
    def bits_dict(self):
        return {self.name: self.bits}

    def copy(self):
        return Bits(self,
                    decoder=self.decoder,
                    random=self.random,
                    name=self.name)

    def rotate(self, num, inplace=False):
        bits = self if inplace else self.copy()
        num = num % len(self)
        bits[:] = bits[-num:] + bits[:-num]
        return None if inplace else bits

    def mutate(self, probability, inplace=False):
        bits = self if inplace else self.copy()
        if len(self) > 0:
            count = self.random.binomial(len(self), probability)
            if count > 0:
                index = self.random.choice(len(self), count, replace=False)
                for i in index:
                    bits[i] = not bits[i]
                return count if inplace else (count, bits)
        return 0 if inplace else (0, bits)

    def cross(self, inputbits,  probability, inplace=False):
        if len(inputbits) != len(self):
            raise ValueError(
                'self and inputbits cannot have different lentgths')
        bits = self if inplace else self.copy()
        if len(self) > 0:
            count = self.random.binomial(len(self), probability)
            if count > 0:
                index = self.random.choice(len(self), count, replace=False)
                for i in index:
                    bits[i] = inputbits[i]
                return count if inplace else (count, bits)
        return 0 if inplace else (0, bits)

    def neighborhood(self):
        def seti(i):
            bits = self.copy()
            bits[i] = not bits[i]
            return bits
        return [seti(i) for i in range(len(self))]

    def randomize(self, inplace=False):
        bits = self if inplace else self.copy()
        bits.mutate(0.5, inplace=True)
        return None if inplace else bits

    def __repr__(self):
        return f'Bits({self.bits})'

    def __str__(self):
        return self.bits


class CBits(Bits):

    def __new__(cls, _, decoder=None, random=None, name=None):
        return super().__new__(cls, _, decoder, random, name)

    def __init__(self, _, decoder=None, random=None, name=None):
        self.decoder = (cat_decoder
                        if decoder is None else decoder)
        super().__init__(_, decoder, random, name)
        if isinstance(_, int) and _ > 0:
            self[0] = 1

        if sum(self) != 1:
            raise ValueError('The sum of CBits instances must be equals to 1')

    def copy(self):
        return CBits(self,
                     decoder=self.decoder,
                     random=self.random,
                     name=self.name)

    def mutate(self, probability, inplace=False):
        bits = self if inplace else self.copy()
        if len(self) >= 2 and self.random.rand() < probability:
            bits.rotate(self.random.randint(1, len(self)), inplace=True)
            return 1 if inplace else (1, bits)
        return 0 if inplace else (0, bits)

    def cross(self, inputbits,  probability, inplace=False):
        if len(self) != len(inputbits):
            raise ValueError(
                'self bits and inputbits cannot have different lentgths')
        bits = self if inplace else self.copy()
        if len(self) >= 1 and self.random.rand() < probability:
            bits[:] = inputbits[:]
            return 1 if inplace else (1, bits)
        return 0 if inplace else (0, bits)

    def neighborhood(self):
        return [self.rotate(i) for i in range(1, len(self))]

    def randomize(self, inplace=False):
        bits = self if inplace else self.copy()
        prob = (len(self) - 1) / len(self)
        bits.mutate(prob, inplace=True)
        return None if inplace else bits

    def __repr__(self):
        return f'CBits({self.bits})'

    def __str__(self):
        return self.bits


class ndBits(tuple):
    def __new__(cls, *args, copy=True):
        args = [arg.copy() for arg in args] if copy else args
        return super(ndBits, cls).__new__(cls, args)

    @property
    def value(self):
        return [bits.value for bits in self]

    @property
    def bits(self):
        return [bits.bits for bits in self]

    @property
    def value_dict(self):
        return {bits.name: bits.value for bits in self}

    @property
    def bits_dict(self):
        return {bits.name: bits.bits for bits in self}

    def copy(self):
        return ndBits(*self)

    def mutate(self, probability, inplace=False):
        ndbits = self if inplace else self.copy()
        count = 0
        for bits in ndbits:
            count += bits.mutate(probability, inplace=True)
        return count if inplace else (count, ndbits)

    def cross(self, inputndbits, probability, inplace=False):
        if len(self) != len(inputndbits):
            raise ValueError(
                'self ndbits and inputndbits cannot have different lentgths')
        ndbits = self if inplace else self.copy()
        count = 0
        for bits, inputbits in zip(ndbits, inputndbits):
            count += bits.cross(inputbits, probability, inplace=True)
        return count if inplace else (count, ndbits)

    def neighborhood(self):
        neighbors = []
        for i in range(len(self)):
            for bits in self[i].neighborhood():
                ndbits = self.copy()
                ndbits[i][:] = bits[:]
                neighbors.append(ndbits)
        return neighbors

    def __repr__(self):
        return f'ndBits({", ".join(self.bits)})'

    def __str__(self):
        return f'({", ".join(self.bits)})'


class Binary:

    def __init__(self,  nbits, seed=None, encoder=None, decoder=None, name=None, BitsClass=Bits):
        self.nbits = nbits
        self.size = 2 ** self.nbits
        self.dimension = nbits
        self.seed = seed
        self.encoder = ((lambda x: int_encoder(x, nbits))
                        if encoder is None else encoder)
        self.decoder = (int_decoder
                        if decoder is None else decoder)
        self.name = name
        self.BitsClass = BitsClass

    @property
    def seed(self):
        raise AttributeError("seed value is not available")

    @seed.setter
    def seed(self, seed):
        self.random = (seed if isinstance(seed, np.random.RandomState)
                       else np.random.RandomState(seed))

    def new(self, x=None):
        kwargs = dict(decoder=self.decoder,
                      random=self.random,
                      name=self.name)
        if x is None:
            return self.BitsClass(self.nbits, **kwargs).randomize()
        bits = self.BitsClass(x, **kwargs)
        if len(bits) == self.nbits:
            return bits
        raise ValueError('The length of bits must be equals to nbits')

    def encode(self, x):
        return self.BitsClass(self.encoder(x),
                              decoder=self.decoder,
                              random=self.random,
                              name=self.name)

    def __repr__(self):
        return f'Binary({self.nbits})'

    def __str__(self):
        return f'{{0,...,{self.size-1}}}n={self.nbits}'


class catBinary(Binary):
    def __init__(self, items, seed=None, name=None):

        if isinstance(items, set):
            items = list(items)
            items.sort()
            self.keys = tuple(items)
            self.items = {key: key for key in self.keys}
        elif isinstance(items, dict):
            keys = list(items.keys())
            keys.sort()
            self.keys = tuple(keys)
            self.items = items
        else:
            raise ValueError(
                '"items" has to be a set instance or a dict instance')
        nbits = len(self.keys)

        def encoder(x): return cat_encoder(x, self.keys)
        def decoder(x): return cat_decoder(x, self.keys, self.items)

        super().__init__(nbits, seed, encoder, decoder, name, CBits)
        self.size = nbits
        self.dimension = 1 if nbits > 1 else 0

    def __repr__(self):
        return f'catBinary({self.nbits})'

    def __str__(self):
        if len(self.keys) == 1:
            return f'{{{self.keys[0]}}}n=1'
        if len(self.keys) == 2:
            return f'{{{self.keys[0]}, {self.keys[-1]}}}n=2'
        return f'{{{self.keys[0]},...,{self.keys[-1]}}}n={self.nbits}'


class floatBinary(Binary):
    def __init__(self, a, b, nbits=None, digits=None, dmin=None, seed=None, name=None):
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

            def encoder(x): return int_encoder(
                int((self.size - 1) * (x - self.a) / (self.b - self.a)), self.nbits)

            def decoder(x): return (
                self.a + (self.b - self.a) * int_decoder(x) / (self.size-1))

            super().__init__(nbits, seed, encoder, decoder, name)
            self.digits = int(-np.log10((b - a) / (self.size - 1)))
            self.dmin = (b - a) / (self.size - 1)
        else:
            def decoder(_): return self.a
            def encoder(_): return bitarray()

            super().__init__(0, seed, encoder, decoder, name)
            self.digits, self.dmin = 0, 0
        self.a, self.b = a, b

    def __repr__(self):
        return f'floatBinary({self.a}, {self.b}, {self.nbits})'

    def __str__(self):
        if self.a == self.b:
            return f'[{self.a}]n={self.nbits}'
        return f'[{self.a}, {self.b}]n={self.nbits}'


class intBinary(floatBinary):

    def __init__(self, a, b, nbits=None, digits=None, dmin=None, seed=None, name=None):
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
        else:
            nbits = 0
        super().__init__(a, b, nbits, None, None, seed, name)
        self.a, self.b = a, b
        encoder = self.encoder
        self.encoder = lambda x: encoder(round(x))
        decoder = self.decoder
        self.decoder = lambda x: round(decoder(x))

    def __repr__(self):
        return f'intBinary({self.a}, {self.b}, {self.nbits})'

    def __str__(self):
        if self.a == self.b:
            return f'{{{self.a}}}n={self.nbits}'
        if self.a == self.b - 1:
            return f'{{{self.a}, {self.b}}}n={self.nbits}'
        return f'{{{self.a},...,{self.b}}}n={self.nbits}'


class ndBinary(tuple):

    def __new__(cls, *args, seed=None):
        for arg in args:
            if not isinstance(arg, Binary):
                raise ValueError(
                    'The arguments must be Binary class instances.')

        return super(ndBinary, cls).__new__(cls, args)

    def __init__(self, *_, seed=None):
        super().__init__()
        self.seed = seed
        self.dimension = 0
        for binary in self:
            binary.seed = self.random
            self.dimension += binary.dimension

    @property
    def seed(self):
        raise AttributeError('The "seed" value not available.')

    @seed.setter
    def seed(self, seed):
        self.random = (seed if isinstance(seed, np.random.RandomState)
                       else np.random.RandomState(seed))
        for binary in self:
            binary.seed = self.random

    def new(self, x=None):
        if x is not None:
            if len(x) == len(self):
                return ndBits(*[binary.new(x) for x, binary in zip(x, self)], copy=False)
            raise ValueError(
                'The length of "x" must be equals to self length')
        return ndBits(*[binary.new() for binary in self], copy=False)

    def encode(self, x):
        if len(x) == len(self):
            return ndBits(*[binary.encode(x) for x, binary in zip(x, self)], copy=False)
        raise ValueError(
            'The length of "x" must be equals to self length')

    def __repr__(self):
        return f'ndBinary{super().__repr__()}'

    def __str__(self):
        return ' x '.join([str(binary) for binary in self])
