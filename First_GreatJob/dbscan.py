import numpy as np


class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None


    def fit(self, X):
        n = X.shape[0]
        self.labels_ = np.ones(n) * -1
        self.X = X

        current_cluster = 0
        for i in range(n):
            if self.labels_[i] == -1:
                self._fit(X[i], i, current_cluster)
                if self.labels_[i] != -1:
                    current_cluster += 1
 

    def _fit(self, x, index, cluster):
        near_points_mask = np.linalg.norm(self.X - x, axis=1) < self.eps
        near_points = self.X[near_points_mask]

        if len(near_points) < self.min_samples and cluster in np.unique(self.labels_): # may be optimized
            self.labels_[index] = cluster
        elif len(near_points) >= self.min_samples:
            self.labels_[index] = cluster
            for i in np.where(near_points_mask == True)[0]:
                if self.labels_[i] == -1:
                    self._fit(self.X[i], i, cluster)
