import numpy as np
from typing import Callable

def biseccion(f: Callable[[float], float], a: float, b: float, tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Encuentra una raíz de la función continua f en el intervalo [a, b] usando el método de bisección.
    Incluye protección contra discontinuidades (asíntotas).

    Args:
        f (Callable): Función objetivo f(x) a evaluar.
        a (float): Límite inferior del intervalo.
        b (float): Límite superior del intervalo.
        tol (float, opcional): Tolerancia o exactitud deseada. Por defecto 1e-6.
        max_iter (int, opcional): Límite de iteraciones para evitar bucles infinitos. Por defecto 100.

    Returns:
        float: Aproximación de la raíz encontrada.

    Raises:
        ValueError: Si la función no cambia de signo o se detecta una discontinuidad (asíntota).
        RuntimeError: Si se alcanza el máximo de iteraciones sin lograr la tolerancia.
    """
    fa = f(a)
    fb = f(b)

    # 1. Validación inicial rigurosa
    if fa * fb > 0:
        raise ValueError(f"La función no cambia de signo en el intervalo [{a}, {b}]. "
                         f"f(a)={fa:g}, f(b)={fb:g}. No cumple el teorema de Bolzano.")
    
    # 2. Casos donde los extremos ya son raíces exactas
    if fa == 0.0: return float(a)
    if fb == 0.0: return float(b)

    punto_medio = a
    
    # 3. Bucle seguro con límite de iteraciones
    for iteracion in range(max_iter):
        punto_medio = a + (b - a) / 2.0 
        fc = f(punto_medio)

        # 4. Criterio de parada por intervalo
        if abs(b - a) / 2.0 < tol:
            # Verificación de discontinuidad ---
            if abs(fc) > 1.0: 
                 raise ValueError(f"Posible discontinuidad detectada en x ≈ {punto_medio}. "
                                  f"El intervalo es menor que la tolerancia, pero |f(x)| = {abs(fc):g} "
                                  f"es demasiado grande para ser una raíz.")
            return punto_medio, iteracion + 1
            
        # 5. Salida temprana si encontramos la raíz exacta
        if fc == 0.0: 
            return punto_medio, iteracion + 1

        if fa * fc < 0:
            b = punto_medio
        else:
            a = punto_medio
            fa = fc
            
    raise RuntimeError(f"El método no convergió a la tolerancia {tol} tras {max_iter} iteraciones. "
                       f"Última estimación: {punto_medio}")
            
    raise RuntimeError(f"El método no convergió a la tolerancia {tol} tras {max_iter} iteraciones. "
                       f"Última estimación: {punto_medio}")
