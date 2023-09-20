import numpy as np

A = np.array([ [ 1,2,3], [4,5,6]])
B = np.array([ [ 1,2,1], [0.2,0.5,1]])

print(A[1,:]-B[1,:])