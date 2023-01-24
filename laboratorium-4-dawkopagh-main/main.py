##
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

from typing import Union, List, Tuple

def chebyshev_nodes(n:int=10)-> np.ndarray:
    """Funkcja tworząca wektor zawierający węzły czybyszewa w postaci wektora (n+1,)

    Parameters:
    n(int): numer ostaniego węzła Czebyszewa. Wartość musi być większa od 0.

    Results:
    np.ndarray: wektor węzłów Czybyszewa o rozmiarze (n+1,).
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(n, int) or n <= 0:
        return None

    cheb = np.array([])
    for k in range(n+1):
        cheb = np.append(cheb,[np.cos(k*np.pi/n)])

    return cheb


def bar_czeb_weights(n:int=10)-> np.ndarray:
    """Funkcja tworząca wektor wag dla węzłów czybyszewa w postaci (n+1,)

    Parameters:
    n(int): numer ostaniej wagi dla węzłów Czebyszewa. Wartość musi być większa od 0.

    Results:
    np.ndarray: wektor wag dla węzłów Czybyszewa o rozmiarze (n+1,).
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(n, int) or n <= 0:
        return None

    cheb_weights = np.array([])

    for j in range(n+1):
        if j == 0 or j == n:
            dlt = 0.5
            cheb_weights = np.append(cheb_weights,[(-1)**j * dlt])
        else:
            dlt = 1
            cheb_weights = np.append(cheb_weights,[(-1)**j * dlt])

    return cheb_weights

def  barycentric_inte(xi:np.ndarray,yi:np.ndarray,wi:np.ndarray,x:np.ndarray)-> np.ndarray:
    """Funkcja przprowadza interpolację metodą barycentryczną dla zadanych węzłów xi
        i wartości funkcji interpolowanej yi używając wag wi. Zwraca wyliczone wartości
        funkcji interpolującej dla argumentów x w postaci wektora (n,) gdzie n to dłógość
        wektora n.

    Parameters:
    xi(np.ndarray): węzły interpolacji w postaci wektora (m,), gdzie m > 0
    yi(np.ndarray): wartości funkcji interpolowanej w węzłach w postaci wektora (m,), gdzie m>0
    wi(np.ndarray): wagi interpolacji w postaci wektora (m,), gdzie m>0
    x(np.ndarray): argumenty dla funkcji interpolującej (n,), gdzie n>0

    Results:
    np.ndarray: wektor wartości funkcji interpolujący o rozmiarze (n,).
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if all(isinstance(i, np.ndarray) for i in [xi, yi, wi, x]):
        if xi.shape != yi.shape or yi.shape != wi.shape:
            return None
        else:
            Y = np.array([])
            for i in x:
                if i in xi:
                    Y = np.append(Y, yi[np.where(xi == 1)[0][0]])
                else:
                    L = wi/(i - xi)
                    Y = np.append(Y, yi @ L / sum(L))
            return Y
    else:
        return None

def L_inf(xr:Union[int, float, List, np.ndarray],x:Union[int, float, List, np.ndarray])-> float:
    """Obliczenie normy  L nieskończonośćg.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach biblioteki numpy.

    Parameters:
    xr (Union[int, float, List, np.ndarray]): wartość dokładna w postaci wektora (n,)
    x (Union[int, float, List, np.ndarray]): wartość przybliżona w postaci wektora (n,1)

    Returns:
    float: wartość normy L nieskończoność,
                                    NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(xr, (int, float, List, np.ndarray)) or not isinstance(x,  (int, float, List, np.ndarray)):
        return np.NaN
    if np.shape(xr) != np.shape(x):
        return np.NaN
    if isinstance(xr, int):
        return np.abs(xr - x)
    return max(np.abs(np.array(xr) - np.array(x)))