import numpy as np
from sklearn.neighbors import NearestNeighbors

def construct_neighborhood_graph(data, n_neighbors):
    # Step 1: Construct the neighborhood graph using k-nearest neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(data)
    _, indices = nn.kneighbors(data)
    
    adjacency_matrix = np.zeros((len(data), len(data)))
    for i, neighbors in enumerate(indices):
        adjacency_matrix[i, neighbors] = 1
        adjacency_matrix[neighbors, i] = 1
    
    return adjacency_matrix

def compute_pairwise_geodesic_distances(adjacency_matrix):
    # Utility function to compute pairwise geodesic distances using shortest path
    n = len(adjacency_matrix)
    distances = np.zeros((n, n))
    
    for i in range(n):
        distances[i] = shortest_path(adjacency_matrix, i)
    
    return distances

def shortest_path(adjacency_matrix, source):
    # Utility function to compute the shortest path using Dijkstra's algorithm
    n = len(adjacency_matrix)
    distances = np.full(n, np.inf)
    distances[source] = 0
    visited = np.zeros(n, dtype=bool)
    
    for _ in range(n):
        min_distance = np.inf
        u = -1
        
        for v in range(n):
            if not visited[v] and distances[v] < min_distance:
                min_distance = distances[v]
                u = v
        
        if u == -1:
            break
        
        visited[u] = True
        
        for v in range(n):
            if not visited[v] and adjacency_matrix[u, v] == 1:
                distances[v] = min(distances[v], distances[u] + 1)
    
    return distances

def construct_geodesic_distance_matrix(distances):
    # Step 3: Construct the geodesic distance matrix using Floyd-Warshall algorithm
    n = len(distances)
    geodesic_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            geodesic_matrix[i, j] = floyd_warshall(distances, i, j)
    
    return geodesic_matrix

def floyd_warshall(distances, source, target):
    # Utility function to compute the shortest geodesic distance between two points
    n = len(distances)
    inf = np.inf
    
    dp = np.zeros((n, n))
    dp.fill(inf)
    np.fill_diagonal(dp, 0)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                dp[i, j] = 0
            elif distances[i, j] != 0:
                dp[i, j] = distances[i, j]
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dp[i, j] = min(dp[i, j], dp[i, k] + dp[k, j])
    
    return dp[source, target]

def apply_mds(geodesic_matrix, n_components):
    # Step 4: Apply Multidimensional Scaling (MDS)
    n = len(geodesic_matrix)
    distances_squared = geodesic_matrix ** 2
    
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H.dot(distances_squared).dot(H)
    
    eigvals, eigvecs = np.linalg.eigh(B)
    sorted_indices = np.argsort(eigvals)[::-1]
    
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    
    mds_embedding = eigvecs[:, :n_components] * np.sqrt(eigvals[:n_components])
    
    return mds_embedding


def isomap(data, n_neighbors, n_components):
    # Step 1: Construct the neighborhood graph
    adjacency_matrix = construct_neighborhood_graph(data, n_neighbors)
    
    # Step 2: Compute pairwise geodesic distances
    distances = compute_pairwise_geodesic_distances(adjacency_matrix)
    
    # Step 3: Construct the geodesic distance matrix
    geodesic_matrix = construct_geodesic_distance_matrix(distances)
    
    # Step 4: Apply Multidimensional Scaling (MDS)
    embedding = apply_mds(geodesic_matrix, n_components)
    
    return embedding