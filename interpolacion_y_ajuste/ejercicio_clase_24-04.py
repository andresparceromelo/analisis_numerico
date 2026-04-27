from modelos import Polinomial_simple
from modelos import minimos_cuadrados
import numpy as np  
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([17,19,20,22,23,25,31,32,33,36,37,38,39,41])
y_data = np.array([19,25,32,51,57,71,141,123,187,192,205,252,248,294])

pendiente, b = minimos_cuadrados(x_data, y_data)

P1 = lambda x: pendiente*x+b

x_values = np.linspace(min(x_data), max(x_data), 200)

plt.plot(x_data, y_data, 'pr', label='datos observados')
plt.plot(x_values, P1(x_values), 'g', label = 'MC')

plt.legend()
plt.xlabel('Diametro Pino')
plt.ylabel('Volúmen')
plt.grid()
plt.show()

#-----------------------------------------------------------#
from modelos import Polinomial_simple
import numpy as np  
import matplotlib.pyplot as plt
import sympy as sp

x = sp.symbols('x')

P2 = Polinomial_simple(x_data, y_data)
P2 = sp.lambdify(x, P2)

x_values = np.linspace(min(x_data), max(x_data), 200)
plt.plot(x_data, y_data, 'pr', label='datos observados')
plt.plot(x_values, P2(x_values), 'g', label = 'MC')
print(P2(18))

plt.legend()
plt.xlabel('Diametro Pino')
plt.ylabel('Volúmen')
plt.grid()
plt.show()




