"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiply 2 floats"""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers"""
    return x + y


def neg(x: float) -> float:
    """Negates a number"""
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another"""
    return x < y


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal"""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers"""
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value"""
    return lt(abs(x - y), 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    return 1 / (1 + math.exp(-x))


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    if x >= 0:
        return x
    else:
        return 0


def log(x: float) -> float:
    """Compute the natural logorithm"""
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function"""
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the natural logorithm"""
    return 1 / (x)


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second arg"""
    return inv(x) * d


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -d / x**2


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    if x > 0:
        return d
    else:
        return 0


# ## Task 0.3


# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable"""

    def function(ls: Iterable[float]) -> Iterable[float]:
        return [fn(i) for i in ls]

    return function


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function"""

    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x,  y in zip(ls1,ls2)]

    # return lambda ls1, ls2: (fn(x, y) for x, y in zip(ls1, ls2))

    return apply


def reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], float], float]:
    """Higher-order function that reduces an  iterable to  a single value using a given function"""

    def apply(ls: Iterable[float], start: float) -> float:
        for i in iter(ls):
            start = fn(start, i)
        return start

    return apply


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map"""
    NL = map(neg)
    return NL(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith"""
    AL = zipWith(add)
    return AL(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list using reduce"""
    start = 0
    SL = reduce(add)
    return SL(ls, start)


def prod(ls: Iterable[float]) -> float:
    """Product of all elements in a list using reduce"""
    start = 1
    ML = reduce(mul)
    return ML(ls, start)
