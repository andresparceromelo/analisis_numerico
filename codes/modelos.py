import numpy as np

def Matrix(x_data):
    n = len(x_data)
    A = np.zeros([n,n], float)
    A[0:n, 0] = 1

    for j in range (1,n):
        for i in range(n):
            A[i,j]=A[i,j-1]*x_data[i]
    return A