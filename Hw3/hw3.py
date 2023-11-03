import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
import random

def rbf_kernel(X, Y, sigma2=0.1):
    # Fill in your code for problem 4(a) here
    # X: n_x x d
    # Y: n_y x d
    # Returns K: n_x x n_y

    # const = np.sqrt(np.pi)/(sigma2*2*np.sqrt(2))
    const = 1
    # print("Done const")
    # K = const*distance.cdist(X,Y)^2
    K = const*np.exp(-np.square(distance.cdist(X,Y))/(2*sigma2))

    # print("Done K")
    # K = np.zeros((X.shape[0], Y.shape[0]))
    return K

def sgd(training_samples, training_labels, step_size=1e-3, batch_size=100, n_iters=200):
    C = np.zeros((training_samples.shape[0], 10)) # n_samples x n_classes
    losses = []
    sample_size = training_samples.shape[0]
    for i in tqdm(range(n_iters)):
        # Fill in your code for problem 4(c) here
        #get L indices from 60,000 samples 
        indices = random.sample(range(0,sample_size),batch_size)
        Kxxi = rbf_kernel(training_samples[indices,:], training_samples)
 
        # f = np.sum(0.5*np.linalg.norm(Kxxi@C - training_labels[indices,:],axis = 0))
        df =  Kxxi.T@(Kxxi@C-training_labels[indices,:])
        # df0 = np.sum(Kxxi.T@(Kxxi@C-training_labels[indices,:]), axis=0)
        # df1 = np.sum(Kxxi.T@(Kxxi@C-training_labels[indices,:]), axis=1)

        # print("axis 0", df0.shape),
        # print("axis 1", df1.shape)
        # print(f.shape)
        print(df.shape)

        C += -df*step_size
        loss = np.linalg.norm(Kxxi@C - training_labels[indices,:])
        print(loss)
        losses.append(loss)

    return C, losses

def test_accuracy(training_samples, test_samples, test_labels, C):
    K_hat = rbf_kernel(test_samples, training_samples)
    print("Done K_hat")
    pred_labels = np.argmax(K_hat @ C, axis=1)
    return np.mean(pred_labels == test_labels)

def main():
    training_samples = np.load('hw3/data/training_samples.npy')
    training_labels = np.load('hw3/data/training_labels.npy')
    test_samples = np.load('hw3/data/test_samples.npy')
    test_labels = np.load('hw3/data/test_labels.npy')

    #for test run
    # Ns = 1000
    # training_samples = training_samples[:Ns,:]
    # training_labels = training_labels[:Ns]
    # test_samples = test_samples[:Ns,:]
    # test_labels = test_labels[:Ns]

    C, losses = sgd(training_samples, training_labels)
    print("done sgd")
    test_acc = test_accuracy(training_samples, test_samples, test_labels, C)
    print('Test accuracy: {}'.format(test_acc))

if __name__ == '__main__':
    main()