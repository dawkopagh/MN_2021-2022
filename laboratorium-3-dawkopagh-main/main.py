import numpy
import numpy as np
import scipy
import pickle
import matplotlib
import matplotlib.pyplot as plt

from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[
    int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(v, (int, float, List, np.ndarray)) or not isinstance(v_aprox, (int, float, List, np.ndarray)):
        return np.NaN
    if isinstance(v, (List, np.ndarray)) and isinstance(v_aprox, (List, np.ndarray)) and len(v) != len(v_aprox):
        return np.NaN

    abs_err = abs(np.array(v_aprox) - np.array(v))
    return abs_err


def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[
    int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(v, (int, float, List, np.ndarray)) or not isinstance(v_aprox, (int, float, List, np.ndarray)):
        return np.NaN
    if isinstance(v, (List, np.ndarray)) and isinstance(v_aprox, (List, np.ndarray)) and len(v) != len(v_aprox):
        return np.NaN
    if 0 in np.array(v):
        return np.NaN

    rel_error = abs(np.array(v_aprox) - np.array(v)) / abs(np.array(v))
    return rel_error


def p_diff(n: int, c: float) -> float:
    """Funkcja wylicza wartości wyrażeń P1 i P2 w zależności od n i c.
    Następnie zwraca wartość bezwzględną z ich różnicy.
    Szczegóły w Zadaniu 2.
    
    Parameters:
    n Union[int]: 
    c Union[int, float]: 
    
    Returns:
    diff float: różnica P1-P2
                NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(n, int) or not isinstance(c, (int, float)):
        return np.NaN

    p1 = 2 ** n - 2 ** n + c
    p2 = 2 ** n + c - 2 ** n
    diff = abs(p1 - p2)
    return diff


def exponential(x: Union[int, float], n: int) -> float:
    """Funkcja znajdująca przybliżenie funkcji exp(x).
    Do obliczania silni można użyć funkcji scipy.math.factorial(x)
    Szczegóły w Zadaniu 3.
    
    Parameters:
    x Union[int, float]: wykładnik funkcji ekspotencjalnej 
    n Union[int]: liczba wyrazów w ciągu
    
    Returns:
    exp_aprox float: aproksymowana wartość funkcji,
                     NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(n, int) or n < 0:
        return np.NaN
    if not isinstance(x, (int, float)):
        return np.NaN
    result = 0.0
    for iter in range(n):
        result += (x ** iter) / scipy.math.factorial(iter)
    return result


def coskx1(k: int, x: Union[int, float]) -> float:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 1.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx float: aproksymowana wartość funkcji,
                 NaN w przypadku błędnych danych wejściowych
    """
    if k == 1:
        return np.cos(x)
    if k == 0:
        return 1
    if not isinstance(k, int) or k < 0:
        return np.NaN
    if not isinstance(x, (int, float)):
        return np.NaN
    else:
        return 2 * np.cos(x) * coskx1(k - 1, x) - coskx1(k - 2, x)


def coskx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 2.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx, sinkx float: aproksymowana wartość funkcji,
                        NaN w przypadku błędnych danych wejściowych
    """
    if k == 1:
        return np.cos(x), np.sin(x)
    elif k == 0:
        return 1, 0
    elif x == 0:
        return 1, 0
    elif not isinstance(k, int) or k < 0:
        return np.NaN
    elif not isinstance(x, (int, float)):
        return np.NaN
    elif k > 0:
        return np.cos(x) * coskx2(k - 1, x)[0] - np.sin(x) * coskx2(k - 1, x)[1],\
               np.sin(x) * coskx2(k - 1, x)[0] + np.cos(x) * coskx2(k - 1, x)[1]


def pi(n: int) -> float:
    """Funkcja znajdująca przybliżenie wartości stałej pi.
    Szczegóły w Zadaniu 5.
    
    Parameters:
    n Union[int, List[int], np.ndarray[int]]: liczba wyrazów w ciągu
    
    Returns:
    pi_aprox float: przybliżenie stałej pi,
                    NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(n,int) or n <= 0:
        return np.NaN
    pi_approx = 0.0

    for iter in range(1,n+1):
        pi_approx += 6/(iter**2)
    return np.sqrt(pi_approx)