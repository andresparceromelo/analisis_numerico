import numpy as np

def eliminacion_gaussiana(A, b):
    """
    Solución de un sistema de ecuaciones lineales mediante Eliminación Gaussiana sin pivoteo.
    Solo funciona adecuadamente para matrices con diagonal principal sin ceros.
    
    Parámetros:
    - A: matriz de coeficientes (cuadrada).
    - b: vector de términos independientes.
    
    Retorna:
    - x: vector solución.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    for i in range(n - 1):
        if np.abs(A[i, i]) < 1e-12:
            raise ValueError(f"Pivote cero o muy cercano a cero detectado en la fila {i}. Se requiere pivoteo.")
            
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - factor * A[i, i:]
            b[j] = b[j] - factor * b[i]
            
    if np.abs(A[n-1, n-1]) < 1e-12:
        raise ValueError("El sistema no tiene solución única (matriz singular).")
        
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        
    return x


def eliminacion_gaussiana_pivoteo(A, b):
    """
    Solución de un sistema de ecuaciones lineales mediante Eliminación Gaussiana con pivoteo parcial.
    
    Parámetros:
    - A: matriz de coeficientes (cuadrada).
    - b: vector de términos independientes.
    
    Retorna:
    - x: vector solución.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    for i in range(n - 1):
        # Pivoteo parcial
        max_idx = np.argmax(np.abs(A[i:, i])) + i
        
        if np.abs(A[max_idx, i]) < 1e-12:
            raise ValueError("El sistema no tiene solución única (matriz singular).")

        if max_idx != i:
            A[[i, max_idx]] = A[[max_idx, i]]
            b[[i, max_idx]] = b[[max_idx, i]]

        # Eliminación
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - factor * A[i, i:]
            b[j] = b[j] - factor * b[i]

    if np.abs(A[n-1, n-1]) < 1e-12:
         raise ValueError("El sistema no tiene solución única (matriz singular).")

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x


def jacobi_matricial(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Método de Jacobi utilizando matrices para resolver Ax = b.
    
    Parámetros:
    - A: matriz de coeficientes (cuadrada).
    - b: vector de términos independientes.
    - x0: vector inicial (opcional).
    - tol: tolerancia para el error iterativo.
    - max_iter: número máximo de iteraciones.
    
    Retorna:
    - x: vector solución aproximado.
    - error_final: error de la última iteración.
    - iteracion: número de iteraciones realizadas.
    - errores: lista con el historial de errores.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    if np.any(np.abs(np.diag(A)) < 1e-12):
        raise ValueError("La matriz tiene ceros en la diagonal principal. El método iterativo fallará.")
        
    if x0 is None:
        x0 = np.zeros(n)
    else:
        x0 = np.array(x0, dtype=float)
        
    D = np.diag(np.diag(A))
    L = D - np.tril(A)
    U = D - np.triu(A)
    
    D_inv = np.linalg.inv(D)
    Tj = np.dot(D_inv, L + U)
    Cj = np.dot(D_inv, b)
    
    error = float('inf')
    iteracion = 0
    errores = []
    
    while error > tol and iteracion < max_iter:
        x1 = np.dot(Tj, x0) + Cj
        error = np.max(np.abs(x1 - x0))
        errores.append(error)
        x0 = x1
        iteracion += 1
        
    return x0, error, iteracion, errores


def jacobi_sumas(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Método de Jacobi utilizando sumatorias elemento a elemento para resolver Ax = b.
    
    Parámetros:
    - A: matriz de coeficientes (cuadrada).
    - b: vector de términos independientes.
    - x0: vector inicial (opcional).
    - tol: tolerancia para el error iterativo.
    - max_iter: número máximo de iteraciones.
    
    Retorna:
    - x: vector solución aproximado.
    - error_final: error de la última iteración.
    - iteracion: número de iteraciones realizadas.
    - errores: lista con el historial de errores.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    if np.any(np.abs(np.diag(A)) < 1e-12):
        raise ValueError("La matriz tiene ceros en la diagonal principal. El método iterativo fallará.")
        
    if x0 is None:
        x0 = np.zeros(n)
    else:
        x0 = np.array(x0, dtype=float)
        
    x1 = np.zeros(n)
    error = float('inf')
    iteracion = 0
    errores = []
    
    while error > tol and iteracion < max_iter:
        for i in range(n):
            # Sumatoria excluyendo el término diagonal
            suma = np.dot(A[i, :i], x0[:i]) + np.dot(A[i, i+1:], x0[i+1:])
            x1[i] = (b[i] - suma) / A[i, i]
            
        error = np.max(np.abs(x1 - x0))
        errores.append(error)
        x0 = np.copy(x1)
        iteracion += 1
        
    return x1, error, iteracion, errores


def gauss_seidel_matricial(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Método de Gauss-Seidel utilizando matrices para resolver Ax = b.
    
    Parámetros:
    - A: matriz de coeficientes (cuadrada).
    - b: vector de términos independientes.
    - x0: vector inicial (opcional).
    - tol: tolerancia para el error iterativo.
    - max_iter: número máximo de iteraciones.
    
    Retorna:
    - x: vector solución aproximado.
    - error_final: error de la última iteración.
    - iteracion: número de iteraciones realizadas.
    - errores: lista con el historial de errores.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    if np.any(np.abs(np.diag(A)) < 1e-12):
        raise ValueError("La matriz tiene ceros en la diagonal principal. El método iterativo fallará.")
        
    if x0 is None:
        x0 = np.zeros(n)
    else:
        x0 = np.array(x0, dtype=float)
        
    D = np.diag(np.diag(A))
    L = D - np.tril(A)
    U = D - np.triu(A)
    
    D_L_inv = np.linalg.inv(D - L)
    Tg = np.dot(D_L_inv, U)
    Cg = np.dot(D_L_inv, b)
    
    error = float('inf')
    iteracion = 0
    errores = []
    
    while error > tol and iteracion < max_iter:
        x1 = np.dot(Tg, x0) + Cg
        error = np.max(np.abs(x1 - x0))
        errores.append(error)
        x0 = x1
        iteracion += 1
        
    return x0, error, iteracion, errores


def gauss_seidel_sumas(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Método de Gauss-Seidel utilizando sumatorias elemento a elemento para resolver Ax = b.
    
    Parámetros:
    - A: matriz de coeficientes (cuadrada).
    - b: vector de términos independientes.
    - x0: vector inicial (opcional).
    - tol: tolerancia para el error iterativo.
    - max_iter: número máximo de iteraciones.
    
    Retorna:
    - x: vector solución aproximado.
    - error_final: error de la última iteración.
    - iteracion: número de iteraciones realizadas.
    - errores: lista con el historial de errores.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    if np.any(np.abs(np.diag(A)) < 1e-12):
        raise ValueError("La matriz tiene ceros en la diagonal principal. El método iterativo fallará.")
        
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    errores = []
    iteracion = 0
    error = float('inf')
    
    for it in range(1, max_iter + 1):
        x_new = x.copy()
        
        for i in range(n):
            suma1 = np.dot(A[i, :i], x_new[:i])
            suma2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - suma1 - suma2) / A[i, i]
            
        error = np.max(np.abs(x_new - x))
        errores.append(error)
        x = x_new
        iteracion = it
        
        if error < tol:
            break
            
    return x, error, iteracion, errores