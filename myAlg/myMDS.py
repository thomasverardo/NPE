import numpy as np


def distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def mypairwise_distances(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = distance(X[i], X[j])
            D[i, j] = d
            D[j, i] = d
    return D


def myMDS(D, n_components):
    
    n = D.shape[0]
    
    # Compute the Gram matrix G
    G = -0.5 * (D**2)

    # Compute the double centering matrix B
    H = np.eye(n) - (1/n) * np.ones((n, n))
    B = np.dot(np.dot(H, G), H)
    
    eigvals, eigvecs = np.linalg.eigh(B)

    # Sort the eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Select the k largest eigenvalues and eigenvectors
    selected_eigvals = eigvals[:n_components]
    selected_eigvecs = eigvecs[:, :n_components]
    
    return selected_eigvecs * np.sqrt(selected_eigvals), eigvals, eigvecs