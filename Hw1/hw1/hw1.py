"run with python Hw1/hw1/hw1.py"
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

# Load dataset: predict iris species from length/width of sepals/petals
# Read about dataset/loader here:  https://www.kaggle.com/datasets/arshid/iris-flower-dataset/

df = pd.read_csv('Hw1/hw1/IRIS.csv') # load data
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

#input x as a vector of length d
def gradF(x, d=d):
    if len(x) != d:
        print("Error: x is not of length d")
        return "BAD"
    grad_out = np.zeros((d,1))
    neg_term = np.log(sum(np.exp(x)))
    for i in range(len(x)):
        grad_out[i] = np.exp(x[i])-neg_term
    return grad_out

gradF([1,2,3])
    
def negative_log_likelihood_loss(A,X,C):
    ##### ANSWER GOES HERE #####
    loss = 0
    one = np.ones((1,N_train))
    grad = np.zeros(A.shape)
    #This would be an array.

    #finding the loss
    eAX = np.exp(np.dot(A,X_train))

    #This should be a vector of k x N_train
    log_eAX = np.log(np.dot(one.T, eAX))
    loss_first_term = np.dot(log_eAX, np.ones((N_train,1)))

    loss_second_term = np.dot(C_train, np.dot(A,X_train).T)
    loss = loss_first_term - loss_second_term
                
    loss = -loss/N_train

    #finding the gradient of the loss
    grad = np.dot(C_train, X_train.T)


    ##### END ANSWER #####
    
    return (loss,grad)



# Now, let's test our formulas by implementing simple gradient descent
A = np.random.normal(size=(k,d)) # initial guess for A
learningRate = 0.1 # step size for gradient descent

for i in range(100):
    (loss,grad) = negative_log_likelihood_loss(A,X_train,C_train)
    (loss_test,grad_test) = negative_log_likelihood_loss(A,X_test,C_test)
    print('loss',i,'=',loss,', test loss',i,'=',loss_test)
    
    A = A - learningRate*grad