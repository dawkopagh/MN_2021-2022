import numpy as np
import scipy
import pickle

from typing import Union, List, Tuple
from copy import copy


def first_spline(x: np.ndarray, y: np.ndarray):
    """Funkcja wyznaczająca wartości współczynników spline pierwszego stopnia.

    Parametrs:
    x(float): argumenty, dla danych punktów
    y(float): wartości funkcji dla danych argumentów

    return (a,b) - krotka zawierająca współczynniki funkcji linowych"""
    if not isinstance(x, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
        return None
    if np.shape(x) != np.shape(y):
        return None

    num_of_ranges = len(x) - 1
    coeff_a = []
    coeff_b = []

    for rng in range(num_of_ranges):
        if (x[rng+1] - x[rng]) != 0:
            coeff_a.append((y[rng+1] - y[rng]) / (x[rng+1] - x[rng]))
            coeff_b.append((y[rng+1] - coeff_a[rng] * x[rng+1]))

    return np.array(coeff_a), np.array(coeff_b)

def cubic_spline(x0, x, y):
    """
    Interpolate a 1-D function using cubic splines.
      x0 : a float or an 1d-array
      x : (N,) array_like
          A 1-D array of real/complex values.
      y : (N,) array_like
          A 1-D array of real values. The length of y along the
          interpolation axis must be equal to the length of x.

    Implement a trick to generate at first step the cholesky matrice L of
    the tridiagonal matrice A (thus L is a bidiagonal matrice that
    can be solved in two distinct loops).

    additional ref: www.math.uh.edu/~jingqiu/math4364/spline.pdf
    """
    x = np.asfarray(x)
    y = np.asfarray(y)

    # remove non finite values
    # indexes = np.isfinite(x)
    # x = x[indexes]
    # y = y[indexes]

    # check if sorted
    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

    size = len(x)

    xdiff = np.diff(x)
    ydiff = np.diff(y)

    # allocate buffer matrices
    Li = np.empty(size)
    Li_1 = np.empty(size-1)
    z = np.empty(size)

    # fill diagonals Li and Li-1 and solve [L][y] = [B]
    Li[0] = (2*xdiff[0])**(1/2)
    Li_1[0] = 0.0
    B0 = 0.0 # natural boundary
    z[0] = B0 / Li[0]

    for i in range(1, size-1, 1):
        Li_1[i] = xdiff[i-1] / Li[i-1]
        Li[i] = (2*(xdiff[i-1]+xdiff[i])**(1/2) - Li_1[i-1] * Li_1[i-1])
        Bi = 6*(ydiff[i]/xdiff[i] - ydiff[i-1]/xdiff[i-1])
        z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    i = size - 1
    Li_1[i-1] = xdiff[-1] / Li[i-1]
    Li[i] = (2*xdiff[-1] - Li_1[i-1] * Li_1[i-1])**(1/2)
    Bi = 0.0 # natural boundary
    z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    # solve [L.T][x] = [y]
    i = size-1
    z[i] = z[i] / Li[i]
    for i in range(size-2, -1, -1):
        z[i] = (z[i] - Li_1[i-1]*z[i+1])/Li[i]

    # find index
    index = x.searchsorted(x0)
    np.clip(index, 1, size-1, index)

    xi1, xi0 = x[index], x[index-1]
    yi1, yi0 = y[index], y[index-1]
    zi1, zi0 = z[index], z[index-1]
    hi1 = xi1 - xi0

    # calculate cubic
    f0 = zi0/(6*hi1)*(xi1-x0)**3 + \
         zi1/(6*hi1)*(x0-xi0)**3 + \
         (yi1/hi1 - zi1*hi1/6)*(x0-xi0) + \
         (yi0/hi1 - zi0*hi1/6)*(xi1-x0)
    return f0


def cubic_spline(x, y, tol=1e-100):
    """
    Interpolate using natural cubic splines.

    Generates a strictly diagonal dominant matrix then applies Jacobi's method.

    Returns coefficients:
    b, coefficient of x of degree 1
    c, coefficient of x of degree 2
    d, coefficient of x of degree 3
    """

    if not isinstance(x, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
        return None
    if np.shape(x) != np.shape(y):
        return None
    x = np.array(x)
    y = np.array(y)
    if np.any(np.diff(x) < 0):
        return None

    size = len(x)
    delta_x = np.diff(x)
    delta_y = np.diff(y)

    A = np.zeros(shape=(size, size))
    b = np.zeros(shape=(size, 1))
    A[0, 0] = 1
    A[-1, -1] = 1

    for i in range(1, size - 1):
        A[i, i - 1] = delta_x[i - 1]
        A[i, i + 1] = delta_x[i]
        A[i, i] = 2 * (delta_x[i - 1] + delta_x[i])

        b[i, 0] = 3 * (delta_y[i] / delta_x[i] - delta_y[i - 1] / delta_x[i - 1])


    c = jacobi(A, b, np.zeros(len(A)), tol=tol, n_iterations=300)


    d = np.zeros(shape=(size - 1, 1))
    b = np.zeros(shape=(size - 1, 1))
    for i in range(0, len(d)):
        d[i] = (c[i + 1] - c[i]) / (3 * delta_x[i])
        b[i] = (delta_y[i] / delta_x[i]) - (delta_x[i] / 3) * (2 * c[i] + c[i + 1])

    return b.squeeze(), c.squeeze(), d.squeeze()


def jacobi(A, b, x0, tol, n_iterations=300):
    """
    Iteracyjne rozwiązanie równania Ax=b dla zadanego x0
    Returns:
    x - estymowane rozwiązanie
    """

    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    counter = 0
    x_diff = tol + 1

    while (x_diff > tol) and (counter < n_iterations):
        for i in range(0, n):
            s = 0
            for j in range(0, n):
                if i != j:
                    s += A[i, j] * x_prev[j]

            x[i] = (b[i] - s) / A[i, i]
        counter += 1
        x_diff = (np.sum((x - x_prev) ** 2)) ** 0.5
        x_prev = x.copy()

    return x



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
    if isinstance(xi, np.ndarray) is True and isinstance(yi, np.ndarray) is True and isinstance(wi, np.ndarray) is True and isinstance(x, np.ndarray):
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
