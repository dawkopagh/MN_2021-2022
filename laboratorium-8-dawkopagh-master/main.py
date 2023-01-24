import numpy as np
import scipy as sp
import pickle

from typing import Union, List, Tuple, Optional


def diag_dominant_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Macierz A ma być diagonalnie zdominowana, tzn. wyrazy na przekątnej sa wieksze od pozostałych w danej kolumnie i wierszu
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: macierz diagonalnie zdominowana o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m, int) and m > 0:
        A = np.random.randint(0, 100, (m,m), dtype=int)
        b = np.random.randint(0, 9, (m,1), dtype=int)
        max_in_row_and_col = np.sum(A, axis=0) + np.sum(A, axis=1)
        A = A + np.diag(max_in_row_and_col)
        return A, b
    else:
        return None

def is_diag_dominant(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest diagonalnie zdominowana
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(A, np.ndarray) and len(A.shape) == 2 and A.shape[0] == A.shape[1]:
        A_size = A.shape[0]
        row_col_sum = []
        for row in range(A_size):
            for col in range(A_size):
                sum = 0
                if row != col:
                    sum += np.abs(A[row][col])
                    sum += np.abs(A[col][row])
            row_col_sum.append(sum)
        if all(np.abs(np.diag(A)) > row_col_sum):
            return True
        else:
            return False
    else:
        return None



def symmetric_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: symetryczną macierz o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m, int) and m > 0:
        A = np.random.randint(0, 100, (m, m), dtype=int)
        b = np.random.randint(0, 9, (m, 1), dtype=int)
        A_symmetric = (A + A.T)
        return A_symmetric, b
    else:
        return None


def is_symmetric(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest symetryczna
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(A, np.ndarray) and len(A.shape) == 2 and A.shape[0] == A.shape[1]:
        return np.allclose(A, A.T, 1e-10, 1e-10)



def solve_jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                 epsilon: Optional[float] = 1e-8, maxiter: Optional[int] = 100) -> Tuple[np.ndarray, int]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych
    Parameters:
    A np.ndarray: macierz współczynników
    b np.ndarray: wektor wartości prawej strony układu
    x_init np.ndarray: rozwiązanie początkowe
    epsilon Optional[float]: zadana dokładność
    maxiter Optional[int]: ograniczenie iteracji
    
    Returns:
    np.ndarray: przybliżone rozwiązanie (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    int: iteracja
    """
    if all(isinstance(element, np.ndarray) for element in [A, b, x_init]) and isinstance(epsilon, float) and isinstance(maxiter, int):
        if epsilon > 0 and maxiter > 0 and len(A.shape) == 2 and A.shape[0] == A.shape[1] and b.shape[0] == A.shape[0] and b.shape[0] == x_init.shape[0]:
            Diagonal_A = np.diag(np.diag(A))
            LU = A - Diagonal_A
            x = x_init
            inv_Diagonal = np.diag(1 / np.diag(Diagonal_A))

            for iter in range(maxiter):
                x_next = np.dot(inv_Diagonal, b - np.dot(LU, x))

                if np.linalg.norm(x_next - x) < epsilon:
                    return x_next, iter
                else:
                    x = x_next
            return x, maxiter
        else:
            return None


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


