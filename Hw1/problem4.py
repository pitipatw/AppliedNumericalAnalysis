import numpy as np

def pivotCol(A):
    N = np.shape(A)[1]
    outmatrix = A.copy()
    # go through each row and pivoting the column to the max one
    for i in range(N):
        for j in range(i,N):
            if outmatrix[i,j] < outmatrix[i, j+1] : 
                temp = outmatrix[:,j+1]
                outmatrix[:, j+1] = outmatrix[:,j]
                outmatrix[:, j] = temp
    return outmatrix

            



def my_det(A):
    N = np.shape(A)[1]
    #check for upper or lower triangular, so that the determinant can be found using just scaling and back-sub
    upper_tri = False

    if upper_tri:
        DoSomething = 0
    else: 
        #check if the first member of the pivoting is the larger among other elements in the same row
        # to prevent machine error scaling thingy.

        # E , for tracking
        #LU decomposition.
        for i in range(N):
            val = A[i,0]
            A[i,:] = A[1,0]/val
            for j in range(i+1,N):
                scale = A[j,i] 
                A[j,:] = A[j,:] - A[j,i]*A[i,:]
    return A


