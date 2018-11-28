from smac.epm.gaussian_process.kernels import KernelBase, Gaussian, Cubic
import numpy as np
from numpy.linalg import lstsq


# 首先满足过所有点的条件
# phi * beta + mu * alpha = f (1)
# 其次再尽量满足梯度条件
# phi' * beta + mu' * alpha = g (2)
# 以下是推论：
# 由(1)
# mu * alpha = f - phi * beta
# alpha = mu-1 * (f - phi * beta) (3)
# 由(1), (3)
# phi' * beta + mu' * mu-1 * (f - phi * beta) = g
# phi' * beta + mu' * mu-1 * f - mu' * mu-1 * phi * beta = g
# (phi' + mu' * mu-1 * phi) * beta = g - mu' * mu-1 * f
# 此处beta用linalg.lstsq求解超定方程
# beta = (phi' + mu' * mu-1 * phi)-1 * (g - mu' * mu-1 * f) (4)
# alpha = mu-1 * (f - phi * (phi' + mu' * mu-1 * phi)-1 * (g - mu' * mu-1 * f))
# alpha = mu-1 * (f - phi * beta) (3)

class GaussianProcessRegressor(object):

    # 此处有一个重大bug，如果用Gaussian做kernel2的话，矩阵元素区分度低，条件数奇异值高。
    # 会导致矩阵逆的计算极其不精确，拟合结果很差
    def __init__(self,
                 kernel1: KernelBase = Gaussian,
                 kernel2: KernelBase = Cubic):
        self.X_train = None
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.alpha = None
        self.beta = None

    def fit(self, X, y, grad):

        # 拷贝X的值
        self.X_train = X.copy()

        # 检查输入项
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            raise ValueError("The input is not a 2-D array.")
        if not isinstance(y, np.ndarray) or len(y.shape) != 1:
            raise ValueError("The output is not a 1-D array.")
        # 记录X的维度，分别是采样点个数N和维度D
        N, D = X.shape

        # 初始化kernel矩阵和kernel的梯度矩阵
        phi = self.kernel1.kernel(X)
        # kernel 2视为主kernel
        mu = self.kernel2.kernel(X)
        phi_prime = self.kernel1.gradient(X)
        mu_prime = self.kernel2.gradient(X)

        # 使用cholesky分解计算两个中间的逆
        # self.mu_inv = np.linalg.inv(mu)
        self.mu_inv = lstsq(mu, np.eye(mu.shape[0]), rcond=None)[0]
        # print(self.mu_inv)
        # print(np.dot(self.mu_inv, mu))
        mu_inv_phi = np.dot(self.mu_inv, phi)
        mu_inv_f = np.dot(self.mu_inv, y)
        # self.L = cholesky(mu, lower=True)
        # self.mu_inv_phi = cho_solve((self.L, True), phi)
        # self.mu_inv_f = cho_solve((self.L, True), y)

        # 算出beta左右的系数，求出超定方程
        beta_left = phi_prime + GaussianProcessRegressor._3d_multiply(mu_prime,
                                                                      mu_inv_phi)
        beta_right = grad - GaussianProcessRegressor._3d_multiply(mu_prime,
                                                                  mu_inv_f)
        # 使用最小二乘法计算超定方程的解
        # self.beta = lstsq(beta_left.reshape(N * D, N), beta_right.reshape(N * D, 1), rcond=None)[0]
        self.beta = lstsq(GaussianProcessRegressor._3d_reshape(beta_left),
                          GaussianProcessRegressor._3d_reshape(
                              beta_right), rcond=None)[0]

        # 使用beta的结论算出alpha的值
        alpha_right = y.reshape(-1) - np.dot(phi, self.beta)
        # alpha = cho_solve((self.L, True), alpha_right)
        self.alpha = np.dot(self.mu_inv, alpha_right)

        return self

    def predict(self, X_test, return_cov=False, return_std=False):
        # 首先进行检查
        # 检查输入项
        if not isinstance(X_test, np.ndarray) or len(X_test.shape) != 2:
            raise ValueError("The input is not a 2-D array.")
        if return_cov and return_std:
            raise RuntimeError("return_cov and return_std cannot be both true.")
        # 模型还未被训练过
        if self.beta is None or self.alpha is None:
            raise ValueError("The model has not been fitted.")

        # 初始化kernel矩阵和kernel的梯度矩阵
        phi = self.kernel1.kernel(self.X_train, X_test=X_test)
        mu = self.kernel2.kernel(self.X_train, X_test=X_test)
        # phi = Cubic.kernel(X_test)
        # mu = Gaussian.kernel(X_test)

        # 进行预测
        y_test = np.dot(phi, self.beta) + np.dot(mu, self.alpha)

        # 判断是否返回协方差，标准差
        if return_cov:
            # self.v = cho_solve((self.L, True), mu.T)
            v = np.dot(self.mu_inv, mu.T)
            y_cov = self.kernel2(X_test) - np.dot(mu, v)
            return y_test, y_cov
        # 返回标准差
        elif return_std:
            # 首先计算X_train的kernel2的逆
            # mu_inv = cho_solve((self.L, True), np.eye(self.L.shape[0]))
            # mu_inv = self.mu_inv

            # 计算方差
            y_var = np.diag(self.kernel2.kernel(X_test)).copy()
            y_var -= np.einsum("ij,ij->i", np.dot(mu, self.mu_inv), mu)

            # 如果方差中出现负值，则全部赋值为0
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                y_var[y_var_negative] = 0.0

            return y_test, np.sqrt(y_var)
        # 直接返回预测结果
        else:
            return y_test

    @staticmethod
    def _3d_multiply(A, B):
        """Return the multiplication of 3D matrix and 2D matrix

        Parameters
        ----------
        A : array, shape (n_samples_X, n_samples_X, n_features)

        B : array, shape (n_samples_X, n_samples_X)

        Returns
        -------
        ret : array, shape (n_samples_X, n_samples_X, n_features)
            C = A * B
        """
        # A是N * M * D的矩阵
        if len(A.shape) != 3:
            raise ValueError("The input A is not a 3-D array.")
        N, M, D = A.shape
        # 而B则是M * P的矩阵(P可能是1)
        if len(B.shape) == 1:
            M = B.shape[0]
            P = 1
        elif len(B.shape) == 2:
            M, P = B.shape
        else:
            raise ValueError("The input B is not a 2-D or 1-D array.")

        if P == 1:
            # 新建一个返回的矩阵
            ret = np.zeros((N, D))
            # 对于第i，j个元素
            for i in range(N):
                # 左行乘右列，叠加所有向量
                for j in range(M):
                    ret[i] += A[i, j, :] * B[j]
        else:
            # 新建一个返回的矩阵
            ret = np.zeros((N, P, D))
            # 对于第i，j个元素
            for i in range(N):
                for k in range(P):
                    for j in range(M):
                        # 左行乘右列，叠加所有向量
                        ret[i][k] += A[i, j, :] * B[j, k]

        return ret

    @staticmethod
    def _3d_reshape(A):
        """Return the multiplication of 3D matrix and 2D matrix

        Parameters
        ----------
        A : array, shape (n_samples_X, n_samples_X, n_features) or (n_samples_X. n_features)

        Returns
        -------
        ret : array, shape (n_samples_X * n_features, n_samples_X) or (n_samples_X * n_features)
        A'
        """
        if A.ndim == 3:
            N, M, D = A.shape
            ret = np.zeros((N * D, N))
            for i in range(N):
                for j in range(N):
                    for k in range(D):
                        ret[i * D + k][j] = A[i][j][k]
        elif A.ndim == 2:
            N, D = A.shape
            ret = np.zeros((N * D))
            for i in range(N):
                for j in range(D):
                    ret[i * D + j] = A[i][j]
        else:
            raise ValueError("The input A is not a 3-D or 2-D array.")

        return ret
