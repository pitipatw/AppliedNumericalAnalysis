import numpy as np

A = np.array([[2,3],[4,5]])
B = np.array([[2,3,4],[4,5,6]])

print(np.matmul(A,B))
print(A.dot(B))