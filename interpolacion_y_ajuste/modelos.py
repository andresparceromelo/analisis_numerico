import numpy as np 
import sel as sl
import sympy as sp
x=sp.symbols('x')

def Matrix(x_data):
    n=len(x_data)
    A=np.zeros([n,n], float)
    A[0:n,0]=1
    for j in range(1,n):
        for i in range(n):
            A[i,j]=A[i,j-1]*x_data[i]
    return A

def Polinomial_simple(x_data, y_data):
    A=Matrix(x_data)
    coef=sl.eliminacion_gaussiana(A,y_data)
    P=sum(coef[i]*x**i for i in range(len(x_data)))
    return P

def Lagrange(x_data, y_data): 
    Poly = 0
    n = len(x_data)

    for i in range(n):
        Li = 1
        for j in range(n):
            if j!= i:
                Li *= (x-x_data[j]) / (x_data[i]-x_data[j])
            
        Poly += Li*y_data[i]
    return Poly


def minimos_cuadrados(x_values, y_values):
    n = len(x_values)

    sum_x = sum(x_values)

    sum_y = sum(y_values)

    sum_yx = sum(x_values * y_values)

    sum_x2 = sum(x_values**2)

    sum_y2 = sum(y_values**2)

    m = (n*sum_yx-sum_x*sum_y) / (n*sum_x2-sum_x**2)

    b =  (sum_x2 * sum_y - sum_x * sum_yx) / (n*sum_x2-sum_x**2)

    return m, b