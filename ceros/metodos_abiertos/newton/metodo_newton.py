import sympy as sp

def newton_rapshon(funcion,x0,tolerancia,x):
    derivada_funcion = sp.diff(funcion,x)
    x1 : float = x0 - (funcion.evalf(subs={x:x0})/derivada_funcion.evalf(subs={x: x0}))
    iteraciones = 0
    while abs(x1 - x0) > tolerancia:
        iteraciones += 1
        x0 = x1
        x1 = x0 - (funcion.evalf(subs={x: x0}) / derivada_funcion.evalf(subs={x: x0}))
    return iteraciones, x1