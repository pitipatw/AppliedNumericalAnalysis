import numpy as np

# A1 = np.array([[0.000000000000000000000000000001,2.,3.],[3000000.,2.,1.],[2.,1.,3.]])
# A2 = np.array( [ [0.0000000000000000000000000000001,1],[1,0]])

# def eet(a,b,N):
#     e1 = np.zeros((N,1))
#     e2 = np.zeros((N,1))
    
#     e1[a-1] = 1
#     e2[b-1] = 1
#     return np.identity(N) + np.matmul(e1,e2.T)

# def addAtoB(A,a,b):
#     N = np.shape(A)[0]
#     return eet(b,a,N).dot(A)


# print(eet(2,3,3))
# print(eet(2,3,3).dot(A1))

# print(addAtoB(A1,3,2))


# print("$$$")



# def pivotCol(A):
#     N = np.shape(A)[1]
#     outmatrix = A.copy()
#     # go through each row and pivoting the column to the max one
#     for i in range(N):
#         for j in range(i,N-1):
#             if outmatrix[i,j] < outmatrix[i, j+1] : 
#                 temp = outmatrix[:,j+1].copy()
#                 # print("Temp")
#                 # print(temp)
#                 outmatrix[:, j+1] = outmatrix[:,j]
#                 outmatrix[:, j] = temp
#     return outmatrix

# def my_bad_det(Ainput):
#     A = Ainput.copy()
#     # print("This is A")
#     # print(A)
#     N = np.shape(A)[1]
#     #check for upper or lower triangular, so that the determinant can be found using just scaling and back-sub


#     #check if the first member of the pivoting is the larger among other elements in the same row
#     # to prevent machine error scaling thingy.

#     #loop each row

#     #First, tidy up the matrix, make sure that the big values stays at the diagonal.
#     for i in range(N): 

#         sign = 1 

#         # #find the maximum values in that row. 
#         # idx = np.argmax(np.abs(A[i,:]))
#         # # print("Large col at ", idx)
#         # temp = A[:,idx].copy()
#         # A[:,idx] = A[:,i].copy()
#         # A[:,i] = temp
#         # # print("new A: \n", A)
#         # if idx != i : 
#         #     # print("swap sign")
#         #     sign *= -1 




#         val = A[i,i]
#         # print("val ", val)
#         # print(val)
#         # A[i,:] = A[i,:]/val
#         # print(A[i,:])
#         # loop rows below row i
#         for j in range(i+1,N):
#             scale = A[j,i].copy()/val
#             # print(scale)
#             # print(scale*A[i,:])
#             # print(A[j,:] - scale*A[i,:])
#             temp = A[j,:] - scale*A[i,:]
#             A[j,:] = temp
#             # print(A)
#             # print("mod A \n", A)
#     return sign*A.diagonal().prod()

def my_det(Ainput):
    A = Ainput.copy()
    # print("This is A")
    # print(A)
    N = np.shape(A)[1]
    
    #check for upper or lower triangular, so that the determinant can be found using just scaling and back-sub
    upper_tri = False
    sign = 1 #every time there is column swap, multiply by -1

    #check if the first member of the pivoting is the larger among other elements in the same row
    # to prevent machine error scaling thingy.

    #loop each row

    #First, tidy up the matrix, make sure that the big values stays at the diagonal.
    for i in range(N): 

        #find the maximum values in that row. 
        idx = np.argmax(np.abs(A[i,:]))
        # print("Large col at ", idx)
        temp = A[:,idx].copy()
        A[:,idx] = A[:,i].copy()
        A[:,i] = temp
        # print("new A: \n", A)
        if idx != i : 
            # print("swap sign")
            sign *= -1 




        val = A[i,i]
        # print(val)
        # A[i,:] = A[i,:]/val
        # print(A[i,:])
        # loop rows below row i
        for j in range(i+1,N):
            scale = A[j,i].copy()/val
            # print(scale)
            # print(scale*A[i,:])
            # print(A[j,:] - scale*A[i,:])
            temp = A[j,:] - scale*A[i,:]
            A[j,:] = temp
            # print(A)
            # print("mod A \n", A)
    return sign*A.diagonal().prod()

# print("A1\n", A1)
# print(my_det(A1))
# print(my_bad_det(A1))


# print("A2\n", A2)
# print(my_det(A2))


# print(my_bad_det(A2))

#tester
N = 1000
for i in range(N):
    d = np.random.randint(1,100)
    A = np.random.normal(size=(d,d))
    A_test = np.linalg.det(A)
    A_mine = my_det(A)
    # print(A_test)
    # print(A_mine)
    error = np.abs(A_test - A_mine)/A_test
    if  error > 0.001:
        print("Fail at d = ",d)
        print("error = ",error)

    