import numpy as np
from numpy.linalg import inv


class FactorAnalysis:
    def __init__(self, W, mu, sigma):
        self.W = W.reshape(1, -1).T
        self.mu = mu
        self.sigma = sigma

    def fit(self, X):
        mu, X, Z, X_by_X, Z_by_Z, X_by_Z = self._perform_expectation(X)

        self.mu = mu

        print("problem 1 answer: updated mu = {}, X = {}".format(self.mu, X))
        print("problem 2 answer: Σ^(Z|X) = {}, <z>_n = {}, <zz>_n = {}"
              .format(self._calc_sigma_of_Z(), Z, self._calc_squared_Z(Z)))

        n_samples = X.shape[0]
        print("problem 3 answer: N = {}, <x'x'^T> = {} <zz^T> ={}, <x'<z>^T> = {}"
              .format(n_samples, X_by_X, Z_by_Z, X_by_Z))

        updated_W, updated_sigma = self._perform_maximization(n_samples, X_by_X, Z_by_Z, X_by_Z)
        print("problem 4 answer: W = {}, Σ = {}".format(updated_W, updated_sigma))

        before_lambda_cov = self.W @ self.W.T + self.sigma
        after_lambda_cov = updated_W @ updated_W.T + updated_sigma
        print("problem 5 answer: before = {}, after = {}".format(before_lambda_cov, after_lambda_cov))

        self.W = updated_W
        self.sigma = updated_sigma

    def _perform_expectation(self, X):
        mu = np.mean(X, axis=0)
        X = X - mu
        Z = self._calc_sigma_of_Z() @ self.W.T @ inv(self.sigma) @ X.T
        X_by_X = self._diag(X.T @ X)
        Z_by_Z = Z @ Z.T + (self._calc_sigma_of_Z() * Z.shape[1])
        X_by_Z = X.T @ Z.T
        return mu, X, Z, X_by_X, Z_by_Z, X_by_Z

    def _perform_maximization(self, n_samples, X_by_X, Z_by_Z, X_by_Z):
        updated_W = X_by_Z @ inv(Z_by_Z)
        updated_cov_mat = self._diag((X_by_X - (X_by_Z @ updated_W.T))) / n_samples
        return updated_W, updated_cov_mat

    def _diag(self, A):
        return np.diag(np.diag(A))

    def _calc_sigma_of_Z(self):
        i_size = self.W.shape[1]
        return inv(self.W.T @ inv(self.sigma) @ self.W + np.eye(i_size))

    def _calc_squared_Z(self, Z):
        return np.multiply(Z, Z) + self._calc_sigma_of_Z()


STUDENT_ID = input("please, input a your id:")
print("Your ID is {}".format(STUDENT_ID))

n_3, n_2, n_1, n_0 = [int(sid) for sid in STUDENT_ID[-4:]]
print("n_0 = {}, n_1 = {}, n_2 = {}, n_3 = {}".format(n_0, n_1, n_2, n_3))

x_1 = np.array([n_0,       n_2 + n_3], dtype=np.float)
x_2 = np.array([n_1 + n_3, n_3],       dtype=np.float)
x_3 = np.array([n_0 + n_3, n_1],       dtype=np.float)
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
