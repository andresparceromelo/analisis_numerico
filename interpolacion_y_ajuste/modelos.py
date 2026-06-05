import numpy as np 
import sel as sl
import sympy as sp
import matplotlib.pyplot as plt

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


def escalas_transformacion(x_data, y_data):

    plt.figure(figsize = (12,12), dpi = 50)

    plt.subplot(331)
    plt.scatter(x_data, y_data, label = 'Original data')
    plt.title('Observed data', fontsize = '10', fontweight = 'bold')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    #--------------------------------------------------------------------
    plt.subplot(332)
    plt.scatter(x_data**2, y_data, color = 'purple', label = '$x^2$')
    plt.title('Transformacion $x^2$', fontsize = '10', fontweight = 'bold')
    plt.xlabel('$x^2$')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    #--------------------------------------------------------------------
    plt.subplot(333)
    plt.scatter(x_data**3, y_data, color = 'blue', label = '$x^3$')
    plt.title('Transformacion $x^3$', fontsize = '10', fontweight = 'bold')
    plt.xlabel('$x^3$')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    #--------------------------------------------------------------------
    plt.subplot(334)
    plt.scatter(np.log(x_data), y_data, color = 'green', label = '$logx$')
    plt.title('Transformacion $logx$', fontsize = '10', fontweight = 'bold')
    plt.xlabel('$logx$')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    #--------------------------------------------------------------------
    plt.subplot(335)
    plt.scatter(x_data, 1/y_data, color = 'black', label = r'$\frac{1}{y}$')
    plt.title(r'Transformacion $\frac{1}{y}$', fontsize = '10', fontweight = 'bold')
    plt.xlabel('x')
    plt.ylabel(r'$\frac{1}{y}$')
    plt.grid()
    plt.legend()
    
    #--------------------------------------------------------------------
    plt.subplot(336)
    plt.scatter(x_data, np.sqrt(y_data), color = 'black', label = '$sqrt{y}$')
    plt.title('Transformacion $sqrt{y}$', fontsize = '10', fontweight = 'bold')
    plt.xlabel('x')
    plt.ylabel('$sqrt{y}$')
    plt.grid()
    plt.legend()
    
    #--------------------------------------------------------------------
    plt.subplot(337)
    plt.scatter(x_data, np.log(y_data), color = 'black', label = '$log{y}$')
    plt.title('Transformacion $log{y}$', fontsize = '10', fontweight = 'bold')
    plt.xlabel('x')
    plt.ylabel('$log{y}$')
    plt.grid()
    plt.legend()
    #-----------------------------------------------------------------------------
    plt.subplot(338)
    plt.scatter(np.log(x_data), np.log(y_data), color = 'black', label = '$log{x} vs log{y}$')
    plt.title('Transformacion $log{x} vs log{y}$', fontsize = '10', fontweight = 'bold')
    plt.xlabel('$log{x}$')
    plt.ylabel('log{y}$')
    plt.grid()
    plt.legend()
    
    #-----------------------------------------------------------------------------
    plt.subplot(339)
    plt.scatter(x_data, y_data**2, color = 'black', label = 'y^2')
    plt.title('Transformacion $y^2$', fontsize = '10', fontweight = 'bold')
    plt.xlabel('x')
    plt.ylabel('$y^2$')
    plt.grid()
    plt.legend()
    
    plt.tight_layout()


def coeficiente_determinacion(x_data, y_data):
    n = len(x_data)
    y_bar = sum(y_data) / n
    m,b = minimos_cuadrados(x_data, y_data)
    
    y_gorro = lambda x: m*x+b
    numerador = sum((y_data-y_gorro(x_data))**2)
    denominador = sum((y_data-y_bar) ** 2)
    R2 = 1 - numerador/denominador
    return R2