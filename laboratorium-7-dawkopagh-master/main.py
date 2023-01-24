import numpy as np
import scipy as sp
from scipy import linalg
from  datetime import datetime
import pickle

from typing import Union, List, Tuple


def spare_matrix_Abt(m: int,n: int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,n), wektora b (m,)  i pomocniczego wektora t (m,) zawierających losowe wartości
    Parameters:
    m(int): ilość wierszy macierzy A
    n(int): ilość kolumn macierzy A
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,n) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if all(isinstance(element, int) for element in [m, n]) and m > 0 and n > 0:
        t = np.linspace(0,1,m)
        b = np.cos(4*t)
        A = []
        for row in range(m):
            A_help = [t[row]**inx for inx in range(n)]
            A.append(A_help)
        return np.array(A), b
    else:
        return None



def square_from_rectan(A: np.ndarray, b: np.ndarray):
    """Funkcja przekształcająca układ równań z prostokątną macierzą współczynników na kwadratowy układ równań. Funkcja ma zwrócić nową macierz współczynników  i nowy wektor współczynników
    Parameters:
      A: macierz A (m,n) zawierająca współczynniki równania
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (n,n) i wektorem (n,)
             Jeżeli dane wejściowe niepoprawne funkcja zwraca None
     """
    if all(isinstance(element, np.ndarray) for element in [A, b]):
        if np.shape(A)[0] == np.shape(b)[0]:
            A_new = np.dot(np.transpose(A), A)
            b_new = np.dot(np.transpose(A), b)
            return A_new, b_new
    else:
        return None



def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

      Parameters:
      A: macierz A (m,n) zawierająca współczynniki równania
      x: wektor x (n,) zawierający rozwiązania równania
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania

      Results:
      (float)- wartość normy residuom dla podanych parametrów
      """
    if all(isinstance(element, np.ndarray) for element in [A, x, b]):
        if np.shape(A)[0] == np.shape(b)[0] and np.shape(A)[1] == np.shape(x)[0]:
            Ax = A @ np.transpose(x)
            return np.linalg.norm(Ax - b)

    else:
        return None