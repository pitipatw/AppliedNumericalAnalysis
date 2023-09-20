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

print(X_train)
print(X_test)

"""
train 120 points
test 30 points
"""
A = np.random.normal(size=(k,d)) # initial guess for A

def negative_log_likelihood_loss(A,X,C):

    ##### ANSWER GOES HERE #####
    k,d = np.shape(A)
    N = np.shape(X)[1]
    ones = np.ones((k,N))
    eAX = np.exp(np.matmul(A,X))
    loss1 = 1/N*np.log( np.matmul(np.transpose(ones), eAX)) 
    loss2 = -1/N*np.trace(np.matmul(np.matmul(np.transpose(C),A),X))
    loss = loss1 + loss2 

    T0 = np.exp(np.matmul(A,X))
    
    grad1 = 1/N*(np.matmul(T0,np.matmul(np.diag(np.transpose(ones)/(np.matmul(np.transpose(ones),T0)), np.transpose(X)))))
    grad2 = -1/N*(np.matmul(np.transpose(A),C))
    grad = grad1 + grad2 
    ##### END ANSWER #####
    
    return (loss,grad)


