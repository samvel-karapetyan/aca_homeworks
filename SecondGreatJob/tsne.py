import numpy as np
from sklearn.metrics import pairwise_distances

class TSNE:
    def __init__(
            self, 
            n_components: int = 2,
            perplexity: float = 1,
            n_iter: int = 200,
            learning_rate = 1e-4):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.learning_rate = learning_rate

        self.desired_perplexity = np.log2(self.perplexity)
        self.embedding_ = None
        self.n_samples = None

    def _shannonEntropy(self, X, i, sigma):
        eps = 1e-8
        similarities = np.exp(-np.linalg.norm(X - X[i], axis=1) / (2 * sigma**2))
        conditional_probs = similarities / ((np.sum(similarities) - 1) + eps)

        return -np.sum(conditional_probs * np.log2(conditional_probs + eps)) + conditional_probs[i] * np.log2(conditional_probs[i] + eps)

    def _binarySearch_perplexity(self, X, i):
        n_iter = 100
        tol = 1e-5

        min_sigma = 1e-8
        max_sigma = 1e8
        sigma = 1

        for _ in range(n_iter):
            entropy = self._shannonEntropy(X, i, sigma)
            print(entropy)
            
            if abs(entropy - self.desired_perplexity) < tol:
                break


            if entropy > self.desired_perplexity:
                max_sigma = sigma

            if entropy < self.desired_perplexity:
                min_sigma = sigma

            sigma = (max_sigma + min_sigma) / 2

        return sigma


    def _variancesInit(self, X):
        variances = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            variances[i] = self._binarySearch_perplexity(X, i)

        return variances
    
    def _jointProbabilities(self, X, sigma):
        eps = 1e-8
        dist = pairwise_distances(X)

        similarities = np.exp(-dist / (2 * sigma**2))
        np.fill_diagonal(similarities, 0)
        conditional_probs = similarities / (np.sum(similarities, axis=1) + eps)
        

        return conditional_probs


    def _gradientDescent(self, prob_hd):
        gradient = np.zeros(self.embedding_.shape)

        for _ in range(self.n_iter):
            prob_ld = self._jointProbabilities(self.embedding_, np.ones(self.n_samples) / np.sqrt(2))

            for i in range(self.n_samples):
                gradient[i] = 2 * np.sum((prob_hd[:, i] - prob_ld[:, i] + prob_hd[i] - prob_ld[i]).reshape(-1, 1) * (self.embedding_[i] - self.embedding_), axis=0)
            
            self.embedding_ -= self.learning_rate * gradient # Momentum

    def fit(self, X):
        self.n_samples = X.shape[0]
        self.embedding_ = np.random.random((self.n_samples, self.n_components))

        variances = self._variancesInit(X)

        prob_hd = self._jointProbabilities(X, variances)

        high_dim_prob = (prob_hd + prob_hd.T) / (2 * self.n_samples)

        self._gradientDescent(high_dim_prob)

    def fit_tranform(self, X):
        self.fit(X)
        return self.embedding_