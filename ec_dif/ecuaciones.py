import numpy as np

def euler(f,a,b,h,co):
    n = int((b-a)/h)
    w = [co]
    for i in range(n):
        w.append(w[i] + h * f(a + i *h, w[i]))
    return np.linspace(a,b,n+1),w

def f(t,y):
    return t*np.exp(3*t) - 2*y

def runge4(f,a,b,h,co):
    n = int((b-a)/h)
    w = [co]

    for i in range(n):
        ti = a+i*h
        k1 = h*f(ti, w[i])
        k2 = h * f(ti + 0.5 * h, w[i] + k1 * 0.5)
        k3 = h * f(ti + 0.5 * h, w[i] + k2 * 0.5)
        k4 = h * f(ti + h, w[i] + k3)
        w.append(w[i] + (1/6)* (k1+2*k2+2*k3+k4))
    return np.linspace(a,b,n+1), w