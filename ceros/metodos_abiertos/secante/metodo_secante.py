def secante(funcion, x0, x1, tolerancia):
    iteraciones = 0
    error = 1
    while (error > tolerancia):
        iteraciones += 1
        x2 = x1 - (funcion(x1) * (x0-x1))/(funcion(x0) - funcion(x1))
        error = abs(x2 - x1)
        x0 = x1
        x1 = x2
    
    return iteraciones, x2