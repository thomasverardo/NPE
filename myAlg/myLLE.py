import numpy as np

## LLE algorithm
from sklearn.neighbors import NearestNeighbors


def lle(X, n_components, n_neighbors):
    # Create a NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1).fit(X)
    k = n_neighbors - 1 #This becase the first neighbor is the point itself

    # Query for nearest neighbors
    Q_distances = []
    Q_indices = []
    for i in range(len(X)):
        distance, index = nbrs.kneighbors([X[i]])
        # print("distance: ", distance[0])
        # print("index: ", index[0])
        Q_distances.append(distance[0][1:])
        Q_indices.append(index[0][1:])


    tol = 1e-5



    n = len(X)
    # W = np.zeros((n, n))
    W = []
    k_1 = np.ones((k))
    I = np.ones((k, 1))

    for i in range(n):
        xi = X[i]
        C = []
        for j in range(k):
            xj = X[Q_indices[i][j]]
            C_aux = []
            for m in range(k):
                xm = X[Q_indices[i][m]]
                C_jk =(xi - xj).T @ (xi - xm)
                C_aux.append(C_jk)
            C.append(C_aux)
        C = np.array(C)
        # print(C.shape)
        C = C + tol * np.eye(*C.shape) # Regularization for C
        w = np.linalg.inv(C) @ k_1
        w = w / (k_1.T @ np.linalg.inv(C) @ k_1)

        # Create an 1 x n array that will contain a 0 if xj is not a 
        # neighbour of xi, otherwise it will cointain the weight of x_j
        w_real = np.zeros((1, n))
        np.put(w_real, Q_indices[i], w)
        W.append(list(w_real[0]))

    W = np.array(W)
    print(W.shape)




    I = np.eye(n)
    M = (I - W).T @ (I - W)

    eigvalues, eigvectors = np.linalg.eig(M)
    print(sorted(np.abs(eigvalues)))
    index_ = np.argsort(np.abs(eigvalues))[1:n_components+1]
    lle_data = eigvectors[:, index_]

    return lle_data