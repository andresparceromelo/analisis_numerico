"""
Eliminación Gaussiana con Pivoteo — código de la profesora
===========================================================
Implementación original: Eliminacion_con_Piv(A, b)
Arreglos añadidos para hacerlo completamente funcional:
  - import numpy as np
  - Conversión de A y b a float64 antes de llamar la función
    (evita errores de tipo cuando se usan listas o arreglos enteros).
  - Bloque __main__ con un ejemplo de uso y verificación residual.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Función de la profesora (transcripción fiel de la imagen)
# ──────────────────────────────────────────────────────────────────────────────

def Eliminacion_con_Piv(A, b):
    """
    Resuelve el sistema Ax = b mediante eliminación gaussiana con
    pivoteo simple (intercambio de filas cuando el pivote es cero).

    Parameters
    ----------
    A : np.ndarray, shape (n, n)  — Matriz de coeficientes (float).
    b : np.ndarray, shape (n,)    — Vector de términos independientes (float).

    Returns
    -------
    x_sol : np.ndarray, shape (n,) — Vector solución del sistema.
    """
    n = len(b)

    # Construir la matriz aumentada [A | b] insertando b como última columna
    A2 = np.insert(A, A.shape[1], b, 1)

    # Vector solución inicializado en cero
    x_sol = np.zeros_like(b)

    # — Pivoteo simple: intercambiar filas cuando el elemento diagonal es cero
    for j in range(n):
        if A2[j, j] == 0:
            aux = A2[j, 0:n + 1].copy()
            A2[j, 0:n + 1] = A2[j + 1, 0:n + 1]
            A2[j + 1, 0:n + 1] = aux
        print(f'fila {j}, {A2[j, 0:n + 1]}')

    # — Eliminación hacia adelante (triangularización)
    for j in range(n):
        for i in range(j + 1, n):
            factor = A2[i, j] / A2[j, j]
            A2[i, 0:n + 1] = A2[i, 0:n + 1] - factor * A2[j, 0:n + 1]

    # — Sustitución regresiva
    for k in range(n - 1, -1, -1):
        x_sol[k] = (A2[k, n] - np.dot(A2[k, k + 1:n], x_sol[k + 1:n])) / A2[k, k]

    print(x_sol)
    return x_sol


# ──────────────────────────────────────────────────────────────────────────────
# Bloque de demostración
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sistema de prueba:
    #   2x  +  y  -  z =  8
    #  -3x  -  y  + 2z = -11
    #  -2x  +  y  + 2z = -3
    # Solución exacta: x=2, y=3, z=-1

    # Conversión a float64 para evitar errores de tipo en las operaciones
    A = np.array([
        [ 2,  1, -1],
        [-3, -1,  2],
        [-2,  1,  2],
    ], dtype=float)

    b = np.array([8, -11, -3], dtype=float)

    print("═" * 45)
    print("Sistema: Ax = b")
    print("A =\n", A)
    print("b =", b)
    print("─" * 45)

    solucion = Eliminacion_con_Piv(A.copy(), b.copy())

    # Verificación residual: ‖b - A·x‖ debe ser ≈ 0
    residuo = np.linalg.norm(b - A @ solucion)
    print(f"\nResiduo ‖b - Ax‖ = {residuo:.2e}")
    print("═" * 45)
