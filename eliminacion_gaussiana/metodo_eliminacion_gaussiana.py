"""
Eliminación Gaussiana con Pivoteo Parcial
==========================================
Resuelve sistemas de ecuaciones lineales de la forma Ax = b
mediante el método de eliminación gaussiana con pivoteo parcial
y sustitución regresiva.

Autor  : Implementación senior — análisis numérico
Fecha  : 2026
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def _build_augmented(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Construye la matriz aumentada [A | b].

    Parameters
    ----------
    A : np.ndarray, shape (n, n)  — Matriz de coeficientes.
    b : np.ndarray, shape (n,)    — Vector de términos independientes.

    Returns
    -------
    np.ndarray, shape (n, n+1)   — Matriz aumentada de trabajo (copia flotante).
    """
    return np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])


def _validate_inputs(A: np.ndarray, b: np.ndarray) -> None:
    """
    Valida que A sea cuadrada y compatible con b.

    Raises
    ------
    ValueError  Si las dimensiones no son compatibles.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A debe ser una matriz cuadrada; se recibió shape {A.shape}.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError(
            f"b debe ser un vector de longitud {A.shape[0]}; se recibió shape {b.shape}."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Núcleo del método
# ──────────────────────────────────────────────────────────────────────────────

def _forward_elimination(M: np.ndarray) -> None:
    """
    Aplica eliminación hacia adelante con pivoteo parcial sobre la
    matriz aumentada M en su lugar (in-place).

    Estrategia de pivoteo parcial
    ─────────────────────────────
    En cada paso k se busca, dentro de la columna k (filas k..n-1),
    el elemento de mayor valor absoluto y se intercambia con la fila k.
    Esto reduce la propagación de errores de redondeo.

    Parameters
    ----------
    M : np.ndarray, shape (n, n+1) — Matriz aumentada (se modifica en su lugar).

    Raises
    ------
    ValueError  Si el sistema es singular (pivote nulo tras el intercambio).
    """
    n = M.shape[0]

    for k in range(n):
        # — Pivoteo parcial: encontrar el índice de la fila con el mayor |valor|
        pivot_row = k + np.argmax(np.abs(M[k:, k]))

        if np.isclose(M[pivot_row, k], 0.0):
            raise ValueError(
                f"El sistema es singular o casi-singular en la columna {k}: "
                "no existe solución única."
            )

        # — Intercambiar fila k con la fila del pivote (si son distintas)
        if pivot_row != k:
            M[[k, pivot_row]] = M[[pivot_row, k]]

        # — Eliminar los elementos por debajo del pivote
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]    # multiplicador de eliminación
            M[i, k:] -= factor * M[k, k:]  # actualizar toda la fila i


def _back_substitution(M: np.ndarray) -> np.ndarray:
    """
    Resuelve el sistema triangular superior mediante sustitución regresiva.

    Parameters
    ----------
    M : np.ndarray, shape (n, n+1) — Matriz aumentada en forma triangular superior.

    Returns
    -------
    np.ndarray, shape (n,) — Vector solución x.
    """
    n = M.shape[0]
    x = np.zeros(n)

    # Recorrer desde la última ecuación hasta la primera
    for i in range(n - 1, -1, -1):
        # Sumar las contribuciones de las incógnitas ya calculadas
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:n])) / M[i, i]

    return x


# ──────────────────────────────────────────────────────────────────────────────
# Interfaz pública
# ──────────────────────────────────────────────────────────────────────────────

def eliminacion_gaussiana(
    A: np.ndarray | list,
    b: np.ndarray | list,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """
    Resuelve el sistema lineal Ax = b mediante eliminación gaussiana
    con pivoteo parcial y sustitución regresiva.

    Parameters
    ----------
    A       : array-like (n, n) — Matriz de coeficientes.
    b       : array-like (n,)   — Vector de términos independientes.
    verbose : bool              — Si True, imprime la matriz aumentada en cada
                                  paso y el vector solución final.

    Returns
    -------
    x : np.ndarray, shape (n,)  — Solución del sistema.

    Raises
    ------
    ValueError  Si A no es cuadrada, si las dimensiones no coinciden,
                o si el sistema es singular.

    Examples
    --------
    >>> A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    >>> b = [8, -11, -3]
    >>> x = eliminacion_gaussiana(A, b, verbose=True)
    >>> x
    array([2., 3., -1.])
    """
    # Convertir a arrays numpy y validar
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    _validate_inputs(A, b)

    # Construir la matriz aumentada de trabajo
    M = _build_augmented(A, b)

    if verbose:
        print("═" * 50)
        print("Matriz aumentada inicial [A | b]:")
        print(M, "\n")

    # Fase 1 — Eliminación hacia adelante (triangularización)
    _forward_elimination(M)

    if verbose:
        print("Matriz triangular superior tras eliminación:")
        print(M, "\n")

    # Fase 2 — Sustitución regresiva
    x = _back_substitution(M)

    if verbose:
        print("Solución del sistema x:")
        for idx, val in enumerate(x):
            print(f"  x[{idx}] = {val:.10g}")
        print("═" * 50)

    return x


# ──────────────────────────────────────────────────────────────────────────────
# Bloque de demostración
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sistema de ejemplo:
    #   2x  +  y  -  z =  8
    #  -3x  -  y  + 2z = -11
    #  -2x  +  y  + 2z = -3
    # Solución exacta: x=2, y=3, z=-1

    A = [
        [ 2,  1, -1],
        [-3, -1,  2],
        [-2,  1,  2],
    ]
    b = [8, -11, -3]

    solucion = eliminacion_gaussiana(A, b, verbose=True)

    # Verificación residual: r = b - A·x  (debe ser ≈ 0)
    residuo = np.array(b) - np.array(A) @ solucion
    print(f"\nResiduo ‖b - Ax‖ = {np.linalg.norm(residuo):.2e}")
