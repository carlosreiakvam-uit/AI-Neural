import numpy as np

A = [[-0.5, 0.5],
     [0.5, 1],
     [0, -0.5]]
B = [[1], [-1]]
matA = np.array(A)
matB = np.array(B)
ape = np.matmul(matA, matB)
print(ape)
