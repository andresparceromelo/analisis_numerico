import numpy as np
import math
from graficador import graficar_funcion
from metodo_biseccion import biseccion

# --- Parámetros del problema ---
r = 2.0
L = 8.0
V_obj = 12.5

def mi_arccos(x):
    """
    Calcula el arco coseno utilizando la fórmula exigida por el problema:
    cos^-1(x) = pi/2 - tan^-1(x / sqrt(1 - x^2))
    """
    # Proteger contra la división por cero si x = 1 o x = -1
    if abs(x) >= 1.0:
        if x >= 1.0: return 0.0
        if x <= -1.0: return math.pi
        
    return (math.pi / 2.0) - math.atan(x / math.sqrt(1.0 - x**2))

def f_tanque(h):
    """
    Función de ceros f(h) = V(h) - V_objetivo.
    Buscamos h tal que f(h) = 0.
    """
    # Validaciones físicas: h debe estar entre 0 y 2*r (profundidad no puede salir del tanque)
    if h <= 0:
        # Si h es cercano a 0 o negativo
        return -V_obj # Volumen casi 0, menos el objetivo
    if h >= 2*r:
        h = 2*r - 1e-9 # Ajuste numérico
        
    term_x = (r - h) / r
    
    volumen = (r**2 * mi_arccos(term_x) - (r - h) * math.sqrt(2*r*h - h**2)) * L
    
    return volumen - V_obj

def main():
    # 1. Graficar para hacer el "bosquejo de la gráfica de la función de ceros"
    # Sabemos que h está en [0, 4] porque el radio es 2 (diámetro = 4)
    print("Mostrando la gráfica para analizar dónde está la raíz...")
    
    # Hacemos la gráfica de 0.01 a 3.99 para evitar singulariadades exactas en los bordes
    graficar_funcion(f_tanque, x_inicio=0.01, x_fin=3.99, 
                     titulo="Función de Volumen del Tanque: f(h) = V(h) - 12.5",
                     xlabel="Profundidad (h) [m]", 
                     ylabel="f(h) [m³]")

    # 2. Una vez que vemos la gráfica, podemos elegir el intervalo.
    # A simple vista (y sabiendo cómo es la función del tanque), la raíz debería estar entre 1 y 3.
    # Vamos a usar nuestro método de bisección:
    a = 0.0
    b = 1.0 
    tol = 1e-4

    print(f"\nAplicando el método de bisección en el intervalo [{a}, {b}] con tolerancia {tol}...")
    try:
        raiz, iteraciones = biseccion(f_tanque, a, b, tol)
        print(f"Profundidad requerida (h): {raiz:.5f} m")
        print(f"Iteraciones realizadas: {iteraciones}")
        print(f"Comprobación (error residual): {f_tanque(raiz):.2e} m³")
        
        # Opcional: Graficar resaltando la raíz
        graficar_funcion(f_tanque, x_inicio=0.01, x_fin=3.99, 
                 titulo="Raíz Encontrada",
                 xlabel="Profundidad (h) [m]", 
                 ylabel="f(h) [m³]",
                 puntos_destacados=[(raiz, 0)])
                 
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()
