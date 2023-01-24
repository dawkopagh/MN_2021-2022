import numpy as np
from typing import Union, Callable
import scipy
import scipy.optimize
import matplotlib
import matplotlib.pyplot as plt
import math

def solve_euler(fun: Callable, t_span: np.array, y0: np.array):
    '''
    Funkcja umożliwiająca rozwiązanie układu równań różniczkowych z wykorzystaniem metody Eulera w przód.

    Parameters:
    fun: Prawa strona równania. Podana funkcja musi mieć postać fun(t, y).
    Tutaj t jest skalarem i istnieją dwie opcje dla ndarray y: Może mieć kształt (n,); wtedy fun musi zwrócić array_like z kształtem (n,).
    Alternatywnie może mieć kształt (n, k); wtedy fun musi zwrócić tablicę typu array_like z kształtem (n, k), tj. każda kolumna odpowiada jednej kolumnie w y.
    t_span: wektor czasu dla którego ma zostać rozwiązane równanie
    y0: warunke początkowy równanai o wymiarze (n,)
    Results:
    (np.array): macierz o wymiarze (n,m) zawierająca w wkolumnach kolejne rozwiązania fun w czasie t_span.

    '''
    if not isinstance(y0, list) and not isinstance(y0, np.array):
        d = fun(t_span[0], y0)
        res = [d]
        for inx in range(1, len(t_span)):
            d1 = d + (t_span[inx] - t_span[inx - 1]) * fun(t_span[inx - 1], d)
            res.append(d1)
            d = d1
        return np.array(res)
    else:
        d = np.transpose(y0)
        res = np.zeros((np.shape(y0)[0], np.shape(t_span)[0]))
        res[:,0] = d
        for inx in range(1, len(t_span)):
            d1 = d + (t_span[inx] - t_span[inx - 1]) * np.transpose(fun(d))
            res[:,inx] = d1
            d=d1
        return(res)
