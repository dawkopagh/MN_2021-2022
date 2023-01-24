import numpy as np
import scipy
import pickle
import typing
import math
import types
import pickle 
from inspect import isfunction


from typing import Union, List, Tuple

def fun(x):
    return np.exp(-2*x)+x**2-1

def dfun(x):
    return -2*np.exp(-2*x) + 2*x

def ddfun(x):
    return 4*np.exp(-2*x) + 2


def bisection(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if all(isinstance(element, (int,float)) for element in [a, b]) and callable(f) and isinstance(epsilon, float) and isinstance(iteration, int):
        if f(a) * f(b) < 0:
            if epsilon > 0 and iteration > 0:
                for inx in range(iteration+1):
                    half = (a+b)/2
                    if f(a) * f(half) <= 0:
                        if np.abs(f(half)) < epsilon or np.abs(a-b) < epsilon or inx == iteration:
                            return half, inx
                        b = half

                    elif f(b) * f(half) <= 0:
                        if np.abs(f(half)) < epsilon or np.abs(a-b) < epsilon or inx == iteration:
                            return half, inx
                        a = half
    return None


def secant(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą siecznych.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if all(isinstance(element, (int, float)) for element in [a, b]) and callable(f) and isinstance(epsilon, float) and isinstance(iteration, int):
        if epsilon > 0 and iteration > 0:
            if f(a) * f(b) < 0:
                for inx in range(iteration + 1):
                    help = (f(b)*a - f(a)*b)/(f(b)-f(a))
                    if f(help)*f(a) >= 0:
                        a = help
                    elif f(help)*f(b) >= 0:
                        b = help
                    if np.abs(b-a) < epsilon or np.abs(f(help)) < epsilon or inx == iteration:
                        return help, inx
                return (f(b) * a - f(a) * b) / (f(b) - f(a)), iteration
    return None

def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float], ddf: typing.Callable[[float], float], a: Union[int,float], b: Union[int,float], epsilon: float, iteration: int) -> Tuple[float, int]:
    ''' Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    Parametry: 
    f - funkcja dla której jest poszukiwane rozwiązanie
    df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    a - początek przedziału
    b - koniec przedziału
    epsilon - tolerancja zera maszynowego (warunek stopu)
    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if all(isinstance(i, (int, float)) for i in [a, b]) and isinstance(epsilon, float) and isinstance(iteration, int):
        if all(callable(x) for x in [f, df, ddf]):
            if f(a) * f(b) < 0 and df(a) * df(b) > 0 and ddf(a) * ddf(b) > 0 and df(a) != 0:
                a = (df(a) * a - f(a)) / df(a)
                for inx in range(iteration + 1):
                    help = a - f(a)/df(a)
                    if np.abs(f(help)) < epsilon or np.abs(help - a) < epsilon or inx == iteration:
                        return help, inx
                    a = help
                return help, iteration
