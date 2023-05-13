import numpy as np
from sklearn.cluster import KMeans

class SpectralClustering:
    """Apply clustering to a projection of the normalized Laplacian.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex, or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster, such as when clusters are
    nested circles on the 2D plane.

    When calling ``fit``, an affinity matrix is constructed using either
    a kernel function such the Gaussian (aka RBF) kernel with Euclidean
    distance ``d(X, X)``::

            np.exp(-gamma * d(X,X) ** 2)

    or a k-nearest neighbors connectivity matrix.
    
    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.

    n_components : int, default=None
        Number of eigenvectors to use for the spectral embedding. If None,
        defaults to `n_clusters`.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia. Only used if
        ``assign_labels='kmeans'``.

    gamma : float, default=1.0
        Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
        Ignored for ``affinity='nearest_neighbors'``.

    affinity : str, default='rbf'
        How to construct the affinity matrix.
         - 'rbf': construct the affinity matrix using a radial basis function
           (RBF) kernel.
         - 'nearest_neighbors': construct the affinity matrix by computing a
           graph of nearest neighbors.

    n_neighbors : int, default=10
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method. Ignored for ``affinity='rbf'``.

    graph : str, default=unnormalized
        Using normalized graph Laplasian or unnormalized graph Laplasian.

    Attributes
    ----------
    affinity_matrix_ : array-like of shape (n_samples, n_samples)
        Affinity matrix used for clustering. Available only after calling
        ``fit``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    """

    def __init__(
            self, 
            n_clusters : int = 8, 
            *,
            gamma : float = 1.0,
            n_components : int = None,
            n_init : int = 10,
            affinity : str = "rbf",
            n_neighbors : int = 10,
            graph : str = "unnormalized"
            ):
        self.n_clusters = n_clusters
        self.gamma = gamma

        if n_components:
            self.n_components = n_components
        else:
            self.n_components = self.n_clusters

        self.n_init = n_init
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.graph = graph

        # Initialize attributes:
        self.affinity_matrix_ = None
        self.laplacian_ = None
        self.labels_ = None
        
    def _rbf(self, X):
        """Computes the affinity matrix using the radial basis function (RBF) kernel.

        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features)
        The input data.

        Returns
        -------
            None. The function updates the `affinity_matrix_` attribute of the class.
        """
        for i in range(X.shape[0]):
            distances = np.linalg.norm(X - X[i], axis=1)
            self.affinity_matrix_[i] = np.exp(-self.gamma * distances**2)

    def _nn(self, X):
        """Computes the affinity matrix using the k-nearest neighbors method.

        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features)
        The input data.

        Returns
        -------
            None. The function updates the `affinity_matrix_` attribute of the class.
        """
        for i in range(X.shape[0]):
            distances = np.linalg.norm(X - X[i], axis=1)
            indeces = np.argsort(distances)[:(self.n_neighbors + 1)] # (n_neighbors + 1) nearest neighbors and same datapoint, 
            for j in indeces:
                dist = np.linalg.norm(X[i] - X[j])
                self.affinity_matrix_[i][j] = np.exp(-self.gamma * dist**2)

        self.affinity_matrix_ = (self.affinity_matrix_ + self.affinity_matrix_.T) / 2

    def fit(self, X):
        """Perform spectral clustering on `X` and return cluster labels.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)

        Returns
        -------
            None
        """
        n = X.shape[0] # Count of datapoints

        self.affinity_matrix_ = np.zeros((n, n)) # Initialize affinity matrix

        if self.affinity == "rbf":
            # Compute affinity matrix using RBF kernel
            self._rbf(X)
        elif self.affinity == "nearest_neighbors":
            # Compute affinity matrix using KNN method
            self._nn(X)

        # Degree of vertex is the sum of the weights of the edges connected to it.
        degree_vertices = self.affinity_matrix_.sum(axis=1) * np.eye(n) 

        if self.graph == "normalized":
            self.laplacian_ = degree_vertices - self.affinity_matrix_ # Unnormalized Laplacian: L = D - W
        else:
            self.laplacian_ = np.eye(n) - np.linalg.inv(degree_vertices) @ self.affinity_matrix_ # Normalized Laplacian: L = I - inv(D) * W

        _, eigvectors = np.linalg.eigh(self.laplacian_)

        pricipal_components = eigvectors[:, :self.n_components]

        kms = KMeans(n_clusters=self.n_clusters, n_init=self.n_init)
        kms.fit(pricipal_components)
    
        self.labels_ = kms.labels_

    def fit_predict(self, X):
        """Perform spectral clustering on `X` and return cluster labels.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X)
        return self.labels_