import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import pandas as pd

x = sp.symbols('x') # Creamos la variable simbólica x

def evaluacion(values, f_sym, p_sym):
    f = sp.lambdify(x, f_sym, "numpy")
    P = sp.lambdify(x, p_sym, "numpy")
    
    lista = []  # Lista donde se guardarán los resultados  
    for i in values:  # Recorremos cada valor de x en la lista dada
        lista.append([
            i,                  # Valor de x
            f(i),               # Valor real de la función en x
            P(i),               # Valor aproximado usando el polinomio de Taylor
            abs(f(i) - P(i)),   # Error absoluto |f(x) - P(x)|
            abs((f(i) - P(i)) / f(i)) if f(i) != 0 else 0 # Error relativo
        ]) 
    # Convertimos la lista en un DataFrame para mostrar los resultados organizados
    lista = pd.DataFrame(data=lista, columns=['values_x', 'f(x)', 'P(x)', '|f(x)-P(x)|', 'E_r'])
    return lista  # Retornamos la tabla con los resultados

def calcular_serie_taylor():
    print("\n=== Calculadora de Series de Taylor ===\n")
    
    try:
        # Variable
        var_input = input("Variable (por defecto x): ").strip()
        if var_input == "":
            var_input = "x"
        x = sp.Symbol(var_input)
        
        # Función
        print("\nEjemplo: cos(x), sin(x), exp(x), log(x)")
        func_input = input(f"Ingrese f({var_input}): ").strip()
        
        # Limpiamos el input por si el usuario pone 'sp.' y usamos sympify con transformaciones
        func_input = func_input.replace('sp.', '')
        # Se habilitan transformaciones como la multiplicación implícita (ej. 2x -> 2*x)
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        transformations = standard_transformations + (implicit_multiplication_application,)
        
        f = parse_expr(func_input, transformations=transformations)
        
        # Punto de expansión
        a_input = input("Punto de expansión a (por defecto 0): ").strip()
        a = sp.nsimplify(sp.sympify(a_input)) if a_input != "" else 0
        
        # Orden
        n_input = input("Orden del polinomio (por defecto 5): ").strip()
        n = int(n_input) if n_input != "" else 5
        
        # Serie
        serie = sp.series(f, x, a, n+1)
        polinomio = serie.removeO()
        
        print("\nPolinomio de Taylor (Vista matemática):")
        sp.pprint(polinomio)

        print("\nPolinomio para copiar y pegar:")
        print(polinomio)
        
        # Evaluación
        while True:
            evaluar = input("\n¿Evaluar en un punto? (s/n): ").lower()
            if evaluar != "s":
                break
            val_input = input(f"Valor de {var_input}: ").strip()
            val = sp.nsimplify(sp.sympify(val_input))
            
            valor_real = f.subs(x, val).evalf()
            valor_aprox = polinomio.subs(x, val).evalf()
            error = abs(valor_real - valor_aprox)
            
            print("\nValor real:", valor_real)
            print("Aproximación Taylor:", valor_aprox)
            print("Error:", error)

            # ── Gráfica inmediata ─────────────────────────────────────────────
            graficar(f, polinomio, x, a, n, func_input, val_x=val)

    except Exception as e:
        print("\nError:", e)


def graficar(f, polinomio, x, a, n, label_f, val_x=None):
    """Grafica f(x) y su polinomio de Taylor p(x) con estilo profesional."""
    a_float = float(a)

    # Rango centrado en el punto de expansión
    radio = max(2.5, abs(a_float) + 2)
    x_min, x_max = a_float - radio, a_float + radio

    xs = np.linspace(x_min, x_max, 1200)

    f_num      = sp.lambdify(x, f,         "numpy")
    p_num      = sp.lambdify(x, polinomio, "numpy")

    with np.errstate(all="ignore"):
        val_f = f_num(xs)
        val_p = p_num(xs)
        
        ys_f = np.full_like(xs, val_f, dtype=complex) if np.isscalar(val_f) else np.array(val_f, dtype=complex)
        ys_p = np.full_like(xs, val_p, dtype=complex) if np.isscalar(val_p) else np.array(val_p, dtype=complex)

    # Descartamos partes imaginarias significativas
    mask_f = np.abs(ys_f.imag) < 1e-9
    mask_p = np.abs(ys_p.imag) < 1e-9
    ys_f = np.where(mask_f, ys_f.real, np.nan)
    ys_p = np.where(mask_p, ys_p.real, np.nan)

    # Clipping para que la gráfica no se dispare
    valid_f = ys_f[np.isfinite(ys_f)]
    valid_p = ys_p[np.isfinite(ys_p)]
    
    max_f = np.max(np.abs(valid_f)) if len(valid_f) > 0 else 1.0
    max_p = np.max(np.abs(valid_p)) if len(valid_p) > 0 else 1.0
    clip = 10 * max(max_f, max_p, 1.0)
    
    ys_f = np.clip(ys_f, -clip, clip)
    ys_p = np.clip(ys_p, -clip, clip)

    # ── Estilo ──────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ax.plot(xs, ys_f, color="#58a6ff", linewidth=2.4, label=f"f(x) = {label_f}")
    ax.plot(xs, ys_p, color="#f78166", linewidth=2.0, linestyle="--",
            label=f"p(x) — Taylor orden {n}")

    # Punto de expansión
    try:
        y_a = float(f.subs(x, a).evalf())
        ax.axvline(a_float, color="#3fb950", linewidth=1.2, linestyle=":")
        ax.scatter([a_float], [y_a], color="#3fb950", s=70, zorder=5,
                   label=f"Punto de expansión a = {a_float}")
    except Exception:
        pass

    # Punto evaluado
    if val_x is not None:
        try:
            vx = float(val_x.evalf())
            vy_f = float(f.subs(x, val_x).evalf())
            vy_p = float(polinomio.subs(x, val_x).evalf())
            ax.scatter([vx], [vy_f], color="#f0e68c", s=80, zorder=6,
                       marker="D", label=f"f({vx:.4g}) = {vy_f:.6g}")
            ax.scatter([vx], [vy_p], color="#da70d6", s=80, zorder=6,
                       marker="D", label=f"p({vx:.4g}) = {vy_p:.6g}")
            ax.plot([vx, vx], [vy_f, vy_p], color="#ffa500",
                    linewidth=1.5, linestyle="-", label=f"Error = {abs(vy_f-vy_p):.4g}")
        except Exception:
            pass

    # Ejes y cuadrícula
    ax.axhline(0, color="#8b949e", linewidth=0.8)
    ax.axvline(0, color="#8b949e", linewidth=0.8)
    ax.grid(True, color="#21262d", linewidth=0.8, linestyle="-")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="minor", color="#1c2128", linewidth=0.4)

    # Etiquetas y título
    ax.set_xlabel("x", fontsize=13, color="#c9d1d9")
    ax.set_ylabel("y", fontsize=13, color="#c9d1d9")
    ax.set_title(f"Serie de Taylor — f(x) = {label_f}  (orden {n}, a = {a_float})",
                 fontsize=14, color="#e6edf3", pad=14)
    ax.tick_params(colors="#8b949e", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    legend = ax.legend(fontsize=11, framealpha=0.25,
                       facecolor="#161b22", edgecolor="#30363d",
                       labelcolor="#e6edf3")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


calcular_serie_taylor()
# print(evaluacion([np.sqrt(3)/2, -1, 1], sp.acos(x), 1.62454782038622 - 1.15470053837925*x))