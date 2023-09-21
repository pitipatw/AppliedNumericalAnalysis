import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

# Load dataset: predict iris species from length/width of sepals/petals
# Read about dataset/loader here:  https://www.kaggle.com/datasets/arshid/iris-flower-dataset/
df = pd.read_csv('IRIS.csv') # load data
df['species'] = LabelEncoder().fit_transform(df['species']) # text labels to numbers
X_all = df.drop(['species'],axis=1) # features
y_all = df['species'] # labels
X_train,X_test,y_train,y_test = train_test_split(X_all,y_all,test_size=0.2,random_state=42) # random split

# Reshape our matrices to fit the homework problem's convention
X_train = X_train.T # transpose so *columns* are feature vectors
X_test = X_test.T

d = X_train.shape[0] # feature dimension
N_train = X_train.shape[1] # size of training set
N_test = X_test.shape[1] # size of test set
k = np.max(y_all)+1 # number of classes

# C matrices encode labels as binary vectors (one per column)
C_train = np.zeros((k,N_train))
C_train[y_train,range(N_train)] = 1
C_test = np.zeros((k,N_test))
C_test[y_test,range(N_test)] = 1

# Homework problem 3b:
#    Input: A,X,C (see homework problem)
#    Output: A pair of loss and its gradient matrix
def negative_log_likelihood_loss(A,X,C):
    # print("A shape")
    # print(np.shape(A))
    # print("X shape")
    # print(np.shape(X))
    X = np.array(X)
    ##### ANSWER GOES HERE #####
    k,d = np.shape(A)
    N = np.shape(X)[1]

    # print(k)
    # print(d)
    # print(N)

    # print("::::")
    # print(A)
    # print(X)
    # print(A.dot(X))
    # print(np.matmul(A,X))
    
    eAX = np.exp(np.matmul(A,X))
    loss1 = 1/N*np.log(np.ones((1,k)).dot(eAX)).dot(np.ones((N,1)))[0][0]
    loss2 = -1/N*np.trace(C.T.dot(A).dot(X))
    # print("losses vals")
    # print(loss1)
    # print(loss2)
    # print("This is loss1 shape")
    # print(np.shape(loss1))
    # print("This is loss2 shape")
    # print(np.shape(loss2))
    loss = loss1 + loss2 

    T0 = np.exp(np.matmul(A,X))
    # print("shape of T0 ")
    # print(np.shape(T0)) #k x N

    # print((np.ones((1,N))/(np.ones((1,k)).dot(T0)))[0])
    # print(np.shape(np.diag((np.ones((1,N))/(np.ones((1,k)).dot(T0)))[0])))
    # print(np.shape(np.diag(np.diag((np.ones((1,N))/(np.ones((1,k)).dot(T0)))[0]))))

    # print((np.ones((1,N))/(np.ones((1,k)).dot(T0)).dot(X.T)))

    grad1 = (1/N)*\
        (T0.dot(\
            np.diag((np.ones((1,N))/(np.ones((1,k)).dot(T0)))[0])\
                    .dot(X.T)))
    grad2 = -1/N*(C.dot(X.T))
    # print("This is grad1")
    # print(np.shape(grad1))
    # print("This is grad2")
    # print(np.shape(grad2))
    grad = grad1 + grad2 
    # print("this is grad")
    # print(grad)
    ##### END ANSWER #####
    
    return (loss,grad)

# for j in range(100):
# Now, let's test our formulas by implementing simple gradient descent
A = np.random.normal(size=(k,d)) # initial guess for A
learningRate = 0.1 # step size for gradient descent

for i in range(100):
    # print("#####train#####")
    (loss,grad) = negative_log_likelihood_loss(A,X_train,C_train)
    # print("#####test#####")
    (loss_test,grad_test) = negative_log_likelihood_loss(A,X_test,C_test)
    print('loss',i,'=',loss,', test loss',i,'=',loss_test)

    A = A - learningRate*grad
# print('loss',i,'=',loss,', test loss',i,'=',loss_test)

print("End of problem 3")

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

    #First, tidy up the matrix, make sure that the big values stays at the diagonal.
    for i in range(N): 

        #find the maximum values in that row. 
        idx = np.argmax(np.abs(A[i,:]))
        # print("Large col at ", idx)
        temp = A[:,idx].copy()
        A[:,idx] = A[:,i].copy()
        A[:,i] = temp
 
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

#tester
N = 100
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

