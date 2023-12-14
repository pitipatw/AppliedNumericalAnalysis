import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
# from sklearn.neighbors import NearestNeighbors


"""
Why k = 5 graph is disconnected
Why K-nearest neighbour not symmetry? You might be my cloest one, but from your point of view, I'm really far from you.

"""
def knn_graph(x, k=10):
    ## Code to compute the k-nearest neighbors graph of (n,d) Numpy array x
 
    A = np.zeros((x.shape[0], x.shape[0]))
    # nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(x)
    for i in range(x.shape[0]):
        d = dist.cdist([x[i,:]],x)[0]
        # print(d)
        idx = np.argsort(d)[1:k+1]
        # print(len(idx))
        A[i,idx] = 1
    # A = nbrs.kneighbors_graph(x).toarray()
    np.fill_diagonal(A , 0)
    return A

def conjugate_gradient(A, b, x0, tol=1e-6, max_steps=None):
    # Conjugate gradient code
    # Inputs:
    # A: (n,n) Numpy array
    # b: (n,) Numpy array
    # x0: (n,) Numpy array
    # tol: float
    # max_steps: int
    x = x0
    # print("Ax shape", (A@x).shape)
    # print("b", b.shape)
    r = b - A@x
    # print("R", r.shape)
    v = r
    for k in range(A.shape[0]):
        # print(k)
        if k > 1000:
            print("Too many iterations")
            break
        alpha = v.T@r/(v.T@A@v)
        x = x + alpha*v
        r_old = r
        r = r - alpha*A@v
        if np.linalg.norm(r) < tol*np.linalg.norm(r_old):
            return x
        # print("r shape", r.shape)
        # print("v shape", v.shape)
        # print("A shape", A.shape)
        v = r - r.T@A@v/(v.T@A@v)*v
        # break

    return x.reshape((-1,1))

def main():
    # Load data
    spiral_data = np.load("spiral_data.npy") # (1000,2) Numpy array
    labels_dict = np.load("labels_dict.npy", allow_pickle=True).item()
    n = spiral_data.shape[0]
    label_idx = list(labels_dict.keys())
    label_val = list(labels_dict.values())
    # spiral_data = spiral_data[:5]

    # Construct the k-nearest neighbors graph of the data
    k = 10

    A = knn_graph(spiral_data , k)

    G = nx.Graph(A)
    Adj = nx.adjacency_matrix(G).todense()
    # print(Adj)

    # Set up linear system for graph-based SSL
    #construct L 
    # print(np.sum(A, axis = 0))
    # print(np.sum(A, axis = 1))
    L = np.zeros((n, n))
    L[Adj == 1] = -1 
    np.fill_diagonal(L, -sum(L, 0))
    L11 = np.delete(L, label_idx, 1)
    L11 = np.delete(L11, label_idx, 0)
    L12 = np.delete(L[:,label_idx], label_idx, 0)
    F1 = np.ones((L11.shape[0],1))*10
    F2 = np.array([label_val]).T
    # print("F2shape", F2.shape)
    b = -L12@F2
 
    # Solve the linear system using conjugate_gradient
    # x_opt is the values of each nodes
    x0 = F1
    # print("x0shape",x0.shape)
    x_opt = conjugate_gradient(L11, b, x0)
    print(x_opt)
    #post processing into full x 
    all_idx = [int(i) for i in np.linspace(0,n-1,n)]
    x_full  = np.zeros((n,1))

    # print(len(np.delete(all_idx,label_idx,0)))
    # print(x_full.shape)
    # print(np.delete(all_idx,label_idx,0).shape)
    # print("xopt shape", x_opt.shape)
    x_full[np.delete(all_idx,label_idx,0),:] = x_opt
    x_full[label_idx,0] = label_val
    
    # Plot the solution
    # x_full = x_opt inserted the constrained.\

    plt.figure(figsize=(8, 8))
    plt.title("Input Graph k = "+str(k))
    # nx.draw_networkx_nodes(
    #     G,
    #     spiral_data,
    #     node_size=10,
    #     cmap=plt.cm.Reds_r,
    # )
    nx.draw_networkx_edges(G, spiral_data, alpha=0.5)
    # nx.draw_networkx(G, spiral_data, alpha=0.5)
    nx.draw_networkx_nodes(
        G,
        spiral_data,
        nodelist=all_idx,
        node_size=10,
        node_color=x_full,
        cmap=plt.cm.Reds_r
    )
    plt.savefig(str(k))
    print(np.max(x_opt))
    print(np.min(x_opt))
    plt.show()
if __name__ == "__main__":
    main()