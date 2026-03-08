import numpy as np
import matplotlib.pyplot as plt

def graficar_funcion(f, x_inicio, x_fin, num_puntos=1000, titulo="Gráfica de la Función", xlabel="x", ylabel="f(x)", mostrar_ejes=True, puntos_destacados=None):
    """
    Grafica una función matemática con un estilo profesional.

    Args:
        f (Callable): Función a graficar.
        x_inicio (float): Límite inferior del eje x.
        x_fin (float): Límite superior del eje x.
        num_puntos (int, opcional): Resolución de la gráfica.
        titulo (str, opcional): Título de la gráfica.
        xlabel (str, opcional): Etiqueta del eje x.
        ylabel (str, opcional): Etiqueta del eje y.
        mostrar_ejes (bool, opcional): Si es True, dibuja líneas en x=0 e y=0.
        puntos_destacados (list of tuples, opcional): Lista de puntos (x, y) a resaltar en la gráfica.
    """
    # Configurar estilo 'pro' (estilo limpio de matplotlib)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Crear el subconjunto de puntos x
    # Añadimos un pequeño "epsilon" si queremos evitar evaluación exacta en singularidades, 
    # pero np.linspace es generalmente confiable.
    x = np.linspace(x_inicio, x_fin, num_puntos)
    
    # Evaluar la función de forma segura (vectorizada o iterativa)
    y = np.zeros_like(x)
    for i, val in enumerate(x):
        try:
            y[i] = f(val)
        except Exception:
            y[i] = np.nan # Si la función falla en un punto, ponemos NaN para que no dibuje
            
    # Crear la figura
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    
    # Graficar la línea principal
    ax.plot(x, y, color='#2E86AB', linewidth=2.5, label='f(x)')
    
    # Dibujar los ejes coordenados (cruz cruzando por 0,0) si se solicitó
    if mostrar_ejes:
        ax.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.5)
        if x_inicio <= 0 <= x_fin:
            ax.axvline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.5)
            
    # Dibujar puntos destacados si los hay (ej. raíces)
    if puntos_destacados:
        for px, py in puntos_destacados:
            ax.plot(px, py, marker='o', markersize=8, color='#D34E24', 
                    markeredgecolor='white', markeredgewidth=1.5, zorder=5)
            # Etiqueta para el punto
            ax.annotate(f'({px:.3g}, {py:.3g})', 
                        xy=(px, py), xytext=(10, 10), 
                        textcoords='offset points', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Estilos de textos y grilla
    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=15, color='#333333')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', color='#555555')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', color='#555555')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Añadir sombra o relleno entre la curva y el eje x (opcional, le da un toque muy pro)
    ax.fill_between(x, y, 0, where=(np.isfinite(y)), color='#2E86AB', alpha=0.1)
    
    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     # Prueba del graficador
#     f_prueba = lambda x: x**2 - 4
#     graficar_funcion(f_prueba, -3, 3, titulo="Prueba del Graficador Pro: f(x) = x^2 - 4", 
#                      puntos_destacados=[(-2, 0), (2, 0)])
