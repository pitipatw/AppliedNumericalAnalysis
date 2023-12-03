import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.spatial.distance as dist
import sklearn.manifold
import time

# Compute matrix B(X) for Smacof
def B(X, D):
    """Input: X is a d x n matrix, D is a n x n matrix
       Output: B(X) is a n x n matrix"""
    # FILL IN YOUR CODE FOR COMPUTING THE MATRIX B(X) 
    n = X.shape[1]
    B_X = np.zeros((n,n))
    distances = dist.cdist(np.transpose(X), np.transpose(X))
    for i in range(n):
        for j in range(i+1,n): 
            if i!=j :
                distance = distances[i,j]
                if distance != 0:
                    B_X[i,j] = -D[i,j]/distance

    #B_X is upper triangular without diagonal. 
    #So, fill B_X first 
    B_X = B_X + B_X.T

    #Calculate diagonals of B_X
    for i in range(n):
        B_X[i,i] = -sum(B_X[i,: ])

    # print(B_X)
    return B_X
    
def smacof_iteration(X, D):
    """Input: X is a d x n matrix, D is a n x n matrix
       Output: X_new is a d x n matrix"""
    # FILL IN YOUR CODE FOR ONE ITERATION OF SMACOF
    n = X.shape[1]

    B_X = B(X, D)
    V = 2*n*(np.eye(n) -1/n * np.ones((n,n)))
    X_new = 2*X@np.transpose(B_X)@np.linalg.inv(V)

    return X_new

def obj(X,D):
    distances = dist.cdist(np.transpose(X), np.transpose(X))
    obj = np.sum(np.square(D - distances))
    return obj

def smacof(D, dims, n_iter):
    """Input is n x n pairwise distance matrix D and number of iterations n_iter.
       Output is dims x n matrix of coordinates X."""
    X = np.random.randn(dims, D.shape[0])
    obj_vals = np.zeros(n_iter)
    obj_old = 0
    for i in range(n_iter):

        if np.mod(i,10) == 0 :
            print("Iteration: ", i)

        X = smacof_iteration(X, D)
        obj_val = obj(X,D)
        conv = np.abs(obj_val - obj_old)/obj_val
        if conv <= 0.001:
            print("End at ", i)
            return X, obj_vals[:i]
        obj_vals[i] = obj_val
        # obj_vals.append(obj_val)
    return X, obj_vals

# Load distance matrix
def load_distances():
    D = np.load('dists.npy')
    return D

# Plot X
def plot_embedding(X, obj_vals, X_sk):
    # fig, ax = plt.subplots(2, 2)
    gs = gridspec.GridSpec(2, 2)

    ax1 = plt.subplot(gs[0, 0])
    plt.title("My plot")

    ax2 = plt.subplot(gs[0, 1])
    plt.title("Scikit-learn.manifold.smacof plot")

    ax3 = plt.subplot(gs[1, :])

    ax1.scatter(X[0,:], X[1,:])
    ax1.set_aspect('equal', adjustable='box')
    

    ax2.scatter(X_sk[:,0], X_sk[:,1])
    ax2.set_aspect('equal', adjustable='box')
    



    n_iter = obj_vals.shape[0]
    x2 = np.linspace(1,n_iter, num= n_iter)
    ax3.plot(x2, obj_vals , '.-')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Objective values')

    plt.show()

def main():
    D = load_distances()

    n_iter = 5000
    dims = 2

    start = time.time()
    X_sk, obj_vals2, sk_n_iter= sklearn.manifold.smacof(D, normalized_stress= False, return_n_iter = True)
    end = time.time()
    print("Elapse time: ", end - start)
    print("Elapse time per iteration: ", (end-start)/sk_n_iter)

    print("Sklearn value: ", obj_vals2)

    start = time.time()
    X, obj_vals = smacof(D, dims, n_iter)
    end = time.time()
    print(obj_vals)
    print("My optimal value: ", obj_vals[-1])
    print("Elapse time: ", end - start)
    print("Elapse time per iteration: ", (end-start)/n_iter)
    plot_embedding(X, obj_vals, X_sk)



main()
