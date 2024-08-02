from numpy import e, pi, cos, exp, sqrt


def _sum(x, y):
    """Devuelve la suma de dos valores."""
    return x + y


def _prod(x, y):
    """Devuelve el producto de dos valores."""
    return x * y


def _iter(x, f=lambda x, _: x, cum=_sum):
    """Itera sobre una lista aplicando una función a cada elemento junto con su índice."""
    z = 0.0 if cum == _sum else 1.0
    for i in range(len(x)):
        z = cum(z, f(x[i], i+1))
    return z


def _itercross(x, f=lambda x, y, _: y - x, cum=_sum):
    """Itera sobre una lista aplicando una función a cada par de elementos consecutivos."""
    z = 0.0 if cum == _sum else 1.0
    for i in range(len(x)-1):
        z = cum(z, f(x[i], x[i+1], i+1))
    return z


def sphere(x):
    return _iter(x, lambda x, _: x**2)


def ackley(x):
    n = len(x)
    return (20.0 + e - 20.0 * exp(-0.2 * sqrt(sphere(x) / n))
            - exp(_iter(x, lambda x, _: cos(2.0 * pi * x)) / n))


def griewank(x):
    return (1 + _iter(x, lambda x, _: x**2) / 4000.0
            - _iter(x, lambda x, i: cos(x / sqrt(i)), _prod))


def rastrigin(x):
    return 10 * len(x) + _iter(x, lambda x, _: x**2 - 10 * cos(2 * pi * x))


def rosenbrock(x):
    return _itercross(x, lambda x, y, _: 100 * (y - x**2) ** 2 + (x - 1)**2)


functions = {'Sphere': dict(function=sphere, interval=[-5.12, 5.12]),
             'Ackley': dict(function=ackley, interval=[-30, 30]),
             'Griewank': dict(function=griewank, interval=[-600, 600]),
             'Rastrigin': dict(function=rastrigin, interval=[-5.12, 5.12]),
             'Rosenbrock': dict(function=rosenbrock, interval=[-2.048, 2.048])}
