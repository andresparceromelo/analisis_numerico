from typing import Callable
import sympy as sp  
x = sp.symbols('x')

def biseccion(f: Callable[[float], float], a: float, b: float, tol: float = 1e-6, max_iter: int = 100):
    """
    Encuentra una raíz de f en el intervalo [a, b] usando el método de Bisección.
    Incluye protección contra discontinuidades (asíntotas).

    Parámetros:
    - f       : función objetivo f(x).
    - a       : límite inferior del intervalo.
    - b       : límite superior del intervalo.
    - tol     : tolerancia deseada (por defecto 1e-6).
    - max_iter: número máximo de iteraciones (por defecto 100).

    Retorna:
    - raiz    : aproximación de la raíz encontrada.
    - iteracion: número de iteraciones realizadas.
    - error   : error absoluto del intervalo en la última iteración.

    Lanza:
    - ValueError  : si f no cambia de signo en [a, b] o se detecta una discontinuidad.
    - RuntimeError: si se alcanza max_iter sin converger.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError(
            f"La función no cambia de signo en [{a}, {b}]. "
            f"f(a)={fa:g}, f(b)={fb:g}. No cumple el teorema de Bolzano."
        )

    if fa == 0.0:
        return float(a), 0, 0.0
    if fb == 0.0:
        return float(b), 0, 0.0

    raiz = a
    error = float('inf')

    for iteracion in range(1, max_iter + 1):
        raiz = a + (b - a) / 2.0
        fc = f(raiz)
        error = abs(b - a) / 2.0

        if error < tol:
            if abs(fc) > 1.0:
                raise ValueError(
                    f"Posible discontinuidad en x ≈ {raiz}. "
                    f"|f(x)| = {abs(fc):g} es demasiado grande para ser una raíz."
                )
            return raiz, iteracion, error

        if fc == 0.0:
            return raiz, iteracion, 0.0

        if fa * fc < 0:
            b = raiz
        else:
            a = raiz
            fa = fc

    raise RuntimeError(
        f"El método no convergió a la tolerancia {tol} tras {max_iter} iteraciones. "
        f"Última estimación: {raiz}"
    )


def falsa_posicion(f: Callable[[float], float], a: float, b: float, tol: float = 1e-6, max_iter: int = 100):
    """
    Encuentra una raíz de f en el intervalo [a, b] usando el método de Falsa Posición
    (Regula Falsi).

    Parámetros:
    - f       : función objetivo f(x).
    - a       : límite inferior del intervalo.
    - b       : límite superior del intervalo.
    - tol     : tolerancia deseada (por defecto 1e-6).
    - max_iter: número máximo de iteraciones (por defecto 100).

    Retorna:
    - raiz    : aproximación de la raíz encontrada.
    - iteracion: número de iteraciones realizadas.
    - error   : |f(raiz)| en la última iteración.

    Lanza:
    - ValueError  : si f no cambia de signo en [a, b].
    - RuntimeError: si se alcanza max_iter sin converger.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError(
            f"La función no cambia de signo en [{a}, {b}]. "
            f"f(a)={fa:g}, f(b)={fb:g}. No cumple el teorema de Bolzano."
        )

    raiz = a
    error = float('inf')

    for iteracion in range(1, max_iter + 1):
        raiz = b - fb * (a - b) / (fa - fb)
        fc = f(raiz)
        error = abs(fc)

        if error < tol:
            return raiz, iteracion, error

        if fa * fc < 0:
            b = raiz
            fb = fc
        else:
            a = raiz
            fa = fc

    raise RuntimeError(
        f"El método no convergió a la tolerancia {tol} tras {max_iter} iteraciones. "
        f"Última estimación: {raiz}"
    )


def newton_raphson(f: Callable[[float], float], df: Callable[[float], float], x0: float, tol: float = 1e-6, max_iter: int = 100):
    """
    Encuentra una raíz de f usando el método de Newton-Raphson.
    Recibe la derivada df de forma explícita (como función de Python), sin depender
    de librerías simbólicas, para mantener coherencia con el resto de la librería.

    Parámetros:
    - f       : función objetivo f(x).
    - df      : derivada de f, df(x).
    - x0      : valor inicial de la iteración.
    - tol     : tolerancia deseada (por defecto 1e-6).
    - max_iter: número máximo de iteraciones (por defecto 100).

    Retorna:
    - raiz    : aproximación de la raíz encontrada.
    - iteracion: número de iteraciones realizadas.
    - error   : |x1 - x0| en la última iteración.

    Lanza:
    - ZeroDivisionError: si la derivada es cero en algún punto de la iteración.
    - RuntimeError     : si se alcanza max_iter sin converger.
    """
    raiz = x0
    error = float('inf')

    for iteracion in range(1, max_iter + 1):
        dfx = df(raiz)

        if abs(dfx) < 1e-12:
            raise ZeroDivisionError(
                f"La derivada es cero en x = {raiz}. El método no puede continuar."
            )

        x1 = raiz - f(raiz) / dfx
        error = abs(x1 - raiz)
        raiz = x1

        if error < tol:
            return raiz, iteracion, error

    raise RuntimeError(
        f"El método no convergió a la tolerancia {tol} tras {max_iter} iteraciones. "
        f"Última estimación: {raiz}"
    )



def newton_rapshon2(funcion,x0,tolerancia):
    derivada_funcion = sp.diff(funcion,x)
    x1 : float = x0 - (funcion.evalf(subs={x:x0})/derivada_funcion.evalf(subs={x: x0}))
    iteraciones = 0
    while abs(x1 - x0) > tolerancia:
        iteraciones += 1
        x0 = x1
        x1 = x0 - (funcion.evalf(subs={x: x0}) / derivada_funcion.evalf(subs={x: x0}))
    return x1

def secante(f: Callable[[float], float], x0: float, x1: float, tol: float = 1e-6, max_iter: int = 100):
    """
    Encuentra una raíz de f usando el método de la Secante.

    Parámetros:
    - f       : función objetivo f(x).
    - x0      : primer valor inicial.
    - x1      : segundo valor inicial.
    - tol     : tolerancia deseada (por defecto 1e-6).
    - max_iter: número máximo de iteraciones (por defecto 100).

    Retorna:
    - raiz    : aproximación de la raíz encontrada.
    - iteracion: número de iteraciones realizadas.
    - error   : |x2 - x1| en la última iteración.

    Lanza:
    - ZeroDivisionError: si f(x0) == f(x1) (recta secante horizontal).
    - RuntimeError     : si se alcanza max_iter sin converger.
    """
    raiz = x1
    error = float('inf')

    for iteracion in range(1, max_iter + 1):
        fx0 = f(x0)
        fx1 = f(x1)

        if abs(fx1 - fx0) < 1e-12:
            raise ZeroDivisionError(
                f"f(x0) ≈ f(x1) en la iteración {iteracion}. La secante es horizontal, el método no puede continuar."
            )

        raiz = x1 - fx1 * (x0 - x1) / (fx0 - fx1)
        error = abs(raiz - x1)
        x0 = x1
        x1 = raiz

        if error < tol:
            return raiz, iteracion, error

    raise RuntimeError(
        f"El método no convergió a la tolerancia {tol} tras {max_iter} iteraciones. "
        f"Última estimación: {raiz}"
    )
