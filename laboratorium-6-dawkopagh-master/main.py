import numpy as np
import pickle

from typing import Union, List, Tuple

def random_matrix_Ab(m:int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    Parameters:
    m(int): rozmiar macierzy
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(m, int) or m <= 0:
        return None

    A = np.random.randint(0, 100, size = (m,m))
    b = np.random.randint(0, 100, size = (m, ))
    return A, b

def residual_norm(A:np.ndarray,x:np.ndarray, b:np.ndarray):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

      Parameters:
      A: macierz A (m,m) zawierająca współczynniki równania 
      x: wektor x (m.) zawierający rozwiązania równania 
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania

      Results:
      (float)- wartość normy residuom dla podanych parametrów"""
    if all(isinstance(i, np.ndarray) for i in [A, x, b]):
        if np.shape(x) == np.shape(b) and np.shape(A)[0] == np.shape(A)[1]:
            Ax = A @ np.transpose(x)
            return np.linalg.norm(b - Ax)
    else:
        return None


def log_sing_value(n:int, min_order:Union[int,float], max_order:Union[int,float]):
    """Funkcja generująca wektor wartości singularnych rozłożonych w skali logarytmiczne
    
        Parameters:
         n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
         min_order(int,float): rząd najmniejszej wartości w wektorze wartości singularnych
         max_order(int,float): rząd największej wartości w wektorze wartości singularnych
         Results:
         np.ndarray - wektor nierosnących wartości logarytmicznych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
         """
    if not isinstance(n, int) or not isinstance(min_order, (int, float)) or not isinstance(max_order, (int, float)):
        return None
    if max_order < min_order or n <= 0 or min_order < 0 or max_order < 0:
        return None

    log_sing = np.logspace(min_order, max_order, n)
    return log_sing[::-1]
    
def order_sing_value(n:int, order:Union[int,float] = 2, site:str = 'gre'):
    """Funkcja generująca wektor losowych wartości singularnych (n,) będących wartościami zmiennoprzecinkowymi losowanymi przy użyciu funkcji np.random.rand(n)*10. 
        A następnie ustawiająca wartość minimalną (site = 'low') albo maksymalną (site = 'gre') na wartość o  10**order razy mniejszą/większą.
    
        Parameters:
        n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
        order(int,float): rząd przeskalowania wartości skrajnej
        site(str): zmienna wskazująca stronnę zmiany:
            - site = 'low' -> sing_value[-1] * 10**order
            - site = 'gre' -> sing_value[0] * 10**order
        
        Results:
        np.ndarray - wektor wartości singularnych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
        """
    if all([isinstance(n, int), isinstance(order, (int, float)), isinstance(site, str)]):
        if n > 0:
            sing = np.random.rand(n)*10

            if site == 'low':
                minElement = np.argmin(sing)
                sing[minElement] = sing[minElement] / 10**order

            elif site == 'gre':
                maxElement = np.argmax(sing)
                sing[maxElement] = sing[maxElement] * 10**(order)

            else:
                return None

            sorted_sing = np.sort(sing)
            return np.flip(sorted_sing)
    return None




def create_matrix_from_A(A:np.ndarray, sing_value:np.ndarray):
    """Funkcja generująca rozkład SVD dla macierzy A i zwracająca otworzenie macierzy A z wykorzystaniem zdefiniowanego wektora warości singularnych

            Parameters:
            A(np.ndarray): rozmiarz macierzy A (m,m)
            sing_value(np.ndarray): wektor wartości singularnych (m,)


            Results:
            np.ndarray: macierz (m,m) utworzoną na podstawie rozkładu SVD zadanej macierzy A z podmienionym wektorem wartości singularnych na wektor sing_valu """
    if not isinstance(A, np.ndarray) or not isinstance(sing_value, np.ndarray):
        return None
    if np.shape(A[0]) != np.shape(sing_value):
        return None

    U, _, V = np.linalg.svd(A)
    return np.dot(U * sing_value, V)
