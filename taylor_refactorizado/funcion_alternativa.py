import numpy as np  #Cálculo numérico eficiente (arreglos y funciones matemáticas)
import matplotlib.pyplot as plt  #Generación de gráficos y visualizaciones
import pandas as pd  #Manejo y análisis de datos con DataFrames
import sympy as sp  #Cálculo simbólico (derivadas, series de Taylor, álgebra)
from math import factorial  #Función factorial del módulo matemático estándar


def Taylor_serie(f, x0, n):
    """
    Calcula el polinomio de Taylor de orden n
    de una función simbólica f centrado en x0.
    
    Parámetros:
    f  -> función simbólica (ej: sp.sin(x))
    x0 -> punto donde se centra la serie
    n  -> orden del polinomio
    
    Retorna:
    Polinomio de Taylor expandido
    """
    polinomio = 0  # Inicializamos el polinomio en 0
    for k in range(n + 1): # Recorremos desde k = 0 hasta k = n
        df = sp.diff(f, x, k) # Calculamos la derivada k-ésima de la función
        df_eval = sp.lambdify(x, df) # Convertimos la derivada simbólica en función evaluable
        valor_derivada = df_eval(x0) # Evaluamos la derivada en el punto x0
        # Construimos el término:
        # (f^(k)(x0) / k!) * (x - x0)^k
        termino = (valor_derivada / factorial(k)) * (x - x0)**k
        polinomio += termino # Sumamos el término al polinomio
    return sp.expand(polinomio) # Expandimos el polinomio para simplificarlo

# print(Taylor_serie(sp.acos(x), 0.5, 1))

