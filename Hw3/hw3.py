import numpy as np
from tqdm import tqdm
from scipy.spatial import distance

def rbf_kernel(X, Y, sigma2=0.1):
    # Fill in your code for problem 4(a) here
    # X: n_x x d
    # Y: n_y x d
    # Returns K: n_x x n_y
    return np.zeros((X.shape[0], Y.shape[0]))

def sgd(training_samples, training_labels, step_size=1e-3, batch_size=100, n_iters=200):
    C = np.zeros((training_samples.shape[0], 10)) # n_samples x n_classes
    losses = []

    for i in tqdm(range(n_iters)):
        # Fill in your code for problem 4(c) here
        loss = 0
        losses.append(loss)
    
    return C, losses

def test_accuracy(training_samples, test_samples, test_labels, C):
    K_hat = rbf_kernel(test_samples, training_samples)
    pred_labels = np.argmax(K_hat @ C, axis=1)
    return np.mean(pred_labels == test_labels)

def main():
    training_samples = np.load('data/training_samples.npy')
    training_labels = np.load('data/training_labels.npy')
    test_samples = np.load('data/test_samples.npy')
    test_labels = np.load('data/test_labels.npy')

    C, losses = sgd(training_samples, training_labels)
    test_acc = test_accuracy(training_samples, test_samples, test_labels, C)
    print('Test accuracy: {}'.format(test_acc))

if __name__ == '__main__':
    main()