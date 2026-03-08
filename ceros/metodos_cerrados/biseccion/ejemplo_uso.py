from metodo_biseccion import biseccion
import math

def funcion_x(x):
    return x**2 - 4


def main():
    print("--- Prueba: f(x) = x^2 - 4 ---")
    try:
        a, b, tol = 0, 5, 1e-6
        # Recibimos DOS valores: la raíz y el contador de iteraciones
        resultado, n_iter = biseccion(funcion_x, a, b, tol)
        
        print(f"La raíz es: {resultado:.6f}")
        print(f"Número de iteraciones reales: {n_iter}")
        print(f"Verificación f(root): {funcion_x(resultado):g}")
        
    except Exception as e:
        print(f"Error: {e}")

# if __name__ == "__main__":
#     main()
