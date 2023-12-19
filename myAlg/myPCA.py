import numpy as np


class myPCA:
    
    def __init__(self):
        # self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_variance_ratio = None
        
    def fit(self, data):
        covariance_matrix = np.cov(data.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        sorted_indices = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:,sorted_indices]
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        self.explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

            
    def transform(self, data, n_components=None):
        if n_components is None:
            selected_eigvecs = self.eigenvectors
        else:
            selected_eigvecs = self.eigenvectors[:, :n_components]
        return data.dot(selected_eigvecs)
    
    def fit_transform(self, data, n_components=None):
        self.fit(data)
        return self.transform(data, n_components)
    
    def inverse_transform(self, data):
        return data.dot(self.eigenvectors.T)


def RBF_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))

def poly_kernel(x, y, sigma=2):
    return (x.dot(y) + 1)**sigma

def compute_kernel_matrix(X, kernel, **kwargs):
    N = X.shape[0]
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(X[i], X[j], **kwargs)
    return K

class myKernelPCA:
    
    def __init__(self, kernel, **kwargs):
        self.kwargs = kwargs
        self.K_centered = None
        self.X_low = None
        
        if kernel == 'rbf':
            self.kernel = RBF_kernel
        elif kernel == 'poly':
            self.kernel = poly_kernel
        else:
            raise ValueError('Kernel not supported')
        
    def fit(self, data):
        K = compute_kernel_matrix(data, self.kernel, **self.kwargs)
        
        # Double centering
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        self.K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        
        eigenvalues, eigenvectors = np.linalg.eig(self.K_centered)
        
        sorted_indices = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:,sorted_indices]
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
    def transform(self, data, n_components=None):
        if n_components is None:
            selected_eigvecs = self.eigenvectors
        else:
            selected_eigvecs = self.eigenvectors[:, :n_components]

        projected_data = self.K_centered.dot(selected_eigvecs)
        return projected_data
    
    def fit_transform(self, data, n_components=2):
        self.fit(data)
        return self.transform(data, n_components)
    
    def inverse_transform(self, data):
        return self.kernel(data, self.X_low, **self.kwargs)