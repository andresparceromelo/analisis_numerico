import numpy as np
import matplotlib.pyplot as plt

def euler(f,a,b,h,co):
    n = int((b-a)/h)
    w = [co]
    for i in range(n):
        w.append(w[i] + h * f(a + i *h, w[i]))
    return np.linspace(a,b,n+1),w

def f(t,y):
    return t*np.exp(3*t) - 2*y

# t, w = euler(f,0,1,0.2,0)

plt.scatter()
