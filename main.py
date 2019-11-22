import numpy as np
from numpy.linalg import inv


class FactorAnalysis:
    def __init__(self, W, mu, sigma):
        self.W = W.reshape(1, -1).T
        self.mu = mu
        self.sigma = sigma

    def fit(self, X):
        X, Z, Z_by_Z, X_by_X, X_by_Z = self._perform_expectation(X)
        print("problem 1 answer: updated mu = {}, X = {}".format(Z, X))
        print("problem 2 answer: Σ^(Z|X) = {}, <z>_n = {}, <zz>_n = {}"
              .format(self._calc_sigma_of_Z(),
                      self._calc_squared_latent_mu(Z),
                      Z_by_Z))

        n_samples = X.shape[0]
        print("problem 3 answer: N = {}, <x'x'^T> = {} "
              "<zz^T> ={}, <x'<z>^T> = {}"
              .format(n_samples, X_by_X, Z_by_Z, X_by_Z))

        updated_W, update_cov_mat = self._perform_maximization(n_samples,
                                                           Z_by_Z,
                                                           X_by_X,
                                                           X_by_Z)
        print("problem 4 answer: W = {}, Σ = {}"
              .format(updated_W, update_cov_mat))
        print("problem 5 answer: after = {}, before = {}".format(
              self.sigma, update_cov_mat))

    def _perform_expectation(self, X):
        mean_vec = np.mean(X)
        X = X - mean_vec
        Z = self._calc_sigma_of_Z() @ self.W.T @ inv(self.sigma) @ X.T
        X_by_X = X.T @ X * np.eye(2)
        Z_by_Z = Z @ Z.T + self._calc_sigma_of_Z()
        X_by_Z = X.T @ Z.T
        return X, Z, X_by_X, Z_by_Z, X_by_Z

    def _perform_maximization(self, n_samples, Z_by_Z, X_by_X, X_by_Z):
        updated_W = (X_by_Z.T @ inv(Z_by_Z)).T
        updated_cov_mat = (X_by_X - X_by_Z @ updated_W.T) * np.eye(2) / n_samples
        return updated_W, updated_cov_mat

    def _calc_sigma_of_Z(self):
        return inv(self.W.T @ inv(self.sigma) @ self.W + np.eye(1))

    def _calc_squared_latent_mu(self, Z):
        return np.multiply(Z, Z) + self._calc_sigma_of_Z()


STUDENT_ID = input("please, input a your id:")
print("Your ID is {}".format(STUDENT_ID))

n_3, n_2, n_1, n_0 = [int(sid) for sid in STUDENT_ID[-4:]]
print("n_0 = {}, n_1 = {}, n_2 = {}, n_3 = {}".format(n_0, n_1, n_2, n_3))

x_1 = np.array([n_0,       n_2 + n_3], dtype=np.float)
x_2 = np.array([n_1 + n_3, n_2 + n_3], dtype=np.float)
x_3 = np.array([n_0 + n_3, n_3],       dtype=np.float)
x_4 = np.array([n_2,       n_0 + n_1], dtype=np.float)
x_5 = np.array([n_0 + n_2, n_1 + n_2], dtype=np.float)
print("x_1 = {}, x_2 = {}, x_3 = {}, x_4 = {}, x_5 = {}")


INPUT_X = np.vstack((x_1, x_2, x_3, x_4, x_5))
INIT_W = np.array([1, 0], dtype=np.float)
INIT_MU = np.array([0, 0], dtype=np.float)
INIT_SIGMA = np.matrix([[1, 0], [0, 1]], dtype=np.float)
print("initialized value...: X = {}, W = {}, μ = {}, Σ = {}"
      .format(INPUT_X, INIT_W, INIT_MU, INIT_SIGMA))
   
fac_model = FactorAnalysis(INIT_W, INIT_MU, INIT_SIGMA)
fac_model.fit(INPUT_X)
