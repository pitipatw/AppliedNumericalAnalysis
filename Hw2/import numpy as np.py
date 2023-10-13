import numpy as np
import scipy as sp

A = np.array([1,2,3,4,5,6])
print(A[[2,1,3]])
B= np.delete(A,[0,1,2])
print(A)
print(np.insert(A, [0,1,2], [0,1,2], axis=0))
