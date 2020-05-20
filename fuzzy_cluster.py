import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans


class BaseCluster:
    def __init__(self, n_cluster, m, scale=1.):
        self.n_cluster = n_cluster
        self.m = m
        self.scale = scale

    def _euclidean_distance(self, X, Y=None):
        """
        return the element-wise euclidean distance between X and Y
        :param X: [n_samples_X, n_features]
        :param Y: if None, return the element-wise distance between X and X, else [n_samples_Y, n_features]
        :return: [n_samples_X, n_samples_Y]
        """
        if Y is None:
            Y = X.copy()
        Y = np.expand_dims(np.transpose(Y), 0)
        X = np.expand_dims(X, 2)
        D = np.sum((X - Y)**2, axis=1)
        return np.sqrt(D)

    def predict(self, X, y=None):
        """
        predict membership grad using fuzzy rules
        :param X: [n_samples, n_features]
        :param y: None
        :return: Mem [n_samples, n_clusters]
        """
        X = np.array(X, dtype=np.float64)
        if y is not None:
            y = np.array(y, dtype=np.float64)

        assert hasattr(self, 'variance_'), "Model not fitted yet."
        assert hasattr(self, 'center_'), "Model not fitted yet."
        d = -(np.expand_dims(X, axis=2) - np.expand_dims(self.center_.T, axis=0))**2 \
            / (2 * self.variance_.T)
        d = np.exp(np.sum(d, axis=1))
        d = np.fmax(d, np.finfo(np.float64).eps)
        return d / np.sum(d, axis=1, keepdims=True)

    def fit(self, X, y=None):
        raise NotImplementedError('Function fit is not implemented yet.')


class FuzzyCMeans(BaseCluster):
    def __init__(self, n_cluster, scale=1., m='auto', error=1e-5, tol_iter=200, verbose=0):
        """
        Implantation of fuzzy c-means
        :param n_cluster: number of clusters
        :param m: fuzzy index
        :param error: max error for u_old - u_new to break the iteration
        :param tol_iter: total iteration number
        :param verbose: whether to print loss infomation during iteration
        """
        self.error = error
        self.tol_iter = tol_iter
        self.n_dim = None
        self.verbose = verbose
        self.fitted = False

        super(FuzzyCMeans, self).__init__(n_cluster, m, scale)

    def get_params(self, deep=True):
        return {
            'n_cluster': self.n_cluster,
            'error': self.error,
            'tol_iter': self.tol_iter,
            'scale': self.scale,
            'm': self.m,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        for p, v in params.items():
            setattr(self, p, v)
        return self

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        if y is not None:
            y = np.array(y, dtype=np.float64)

        if self.m == 'auto':
            if min(X.shape[0], X.shape[1]-1) >=3:
                self.m = min(X.shape[0], X.shape[1]-1) / (min(X.shape[0], X.shape[1]-1) - 2)
            else:
                self.m = 2

        N = X.shape[0]
        self.n_dim = X.shape[1]

        # init U
        U = np.random.rand(self.n_cluster, N)

        self.loss_hist = []
        for t in range(self.tol_iter):
            U, V, loss, signal = self._cmean_update(X, U)
            self.loss_hist.append(loss)
            if self.verbose > 0:
                print('[FCM Iter {}] Loss: {:.4f}'.format(t, loss))
            if signal:
                break
        self.fitted = True
        self.center_ = V
        self.train_u = U
        self.variance_ = np.zeros(self.center_.shape)  # n_clusters * n_dim

        for i in range(self.n_dim):
            self.variance_[:, i] = np.sum(
                U * ((X[:, i][:, np.newaxis] - self.center_[:, i].transpose())**2).T, axis=1
            ) / np.sum(U, axis=1)
        self.variance_ *= self.scale
        self.variance_ = np.fmax(self.variance_, np.finfo(np.float64).eps)

        return self

    def _cmean_update(self, X, U):
        old_U = U.copy()
        old_U = np.fmax(old_U, np.finfo(np.float64).eps)
        old_U_unexp = old_U.copy()
        old_U = self.normalize_column(old_U)**self.m

        # compute V
        V = np.dot(old_U, X) / old_U.sum(axis=1, keepdims=True)

        # compute U
        dist = self._euclidean_distance(X, V).T  # n_clusters * n_samples
        dist = np.fmax(dist, np.finfo(np.float64).eps)

        loss = (old_U * dist ** 2).sum()
        dist = dist ** (2/(1-self.m))
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        U = self.normalize_column(dist)
        if np.linalg.norm(U - old_U_unexp) < self.error:
            signal = True
        else:
            signal = False
        return U, V, loss, signal

    def normalize_column(self, U):
        return U/np.sum(U, axis=0, keepdims=True)

    def __str__(self):
        return "FCM"

    def fs_complexity(self):
        return self.n_cluster * self.n_dim


class ESSC(BaseCluster):
    """
    Implementation of the enhanced soft subspace cluster ESSC in the paper "Enhanced soft subspace clustering
    integrating within-cluster and between-cluster information".
    """
    def __init__(self, n_cluster, scale=1., m='auto', eta=0.1, gamma=0.1,
                 error=1e-5, tol_iter=50, verbose=0, init='kmean', sparse_thres=0.0):
        """
        :param n_cluster:
        :param scale:
        :param m:
        :param eta:
        :param gamma:
        :param error:
        :param tol_iter:
        :param verbose:
        :param init:
        :param sparse_thres: percentile for dropping attributes
        """
        super(ESSC, self).__init__(n_cluster, m, scale)
        assert gamma > 0, "gamma must be larger than 0"
        assert (eta < 1) and (eta > 0), "eta must be in the range of [0, 1]"
        self.eta = eta
        self.gamma = gamma
        self.error = error
        self.tol_iter = tol_iter
        self.n_dim = None
        self.verbose = verbose
        self.fitted = False
        self.init_method = init
        self.sparse_thres = sparse_thres

        self.U, self.weight_, self.center_, self.v0 = None, None, None, None

    def get_params(self, deep=True):
        return {
            'n_cluster': self.n_cluster,
            'scale': self.scale,
            'eta': self.eta,
            'gamma': self.gamma,
            'max_iter': self.tol_iter,
            'error': self.error,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        for p, v in params.items():
            setattr(self, p, v)
        return self

    def fit(self, X, y=None):
        """
        :param X: shape: [n_samples, n_features]
        :param y:
        :return:
        """
        X = np.array(X, dtype=np.float64)
        if y is not None:
            y = np.array(y, dtype=np.float64)
        if self.m == 'auto':
            if min(X.shape[0], X.shape[1]-1) >=3:
                self.m = min(X.shape[0], X.shape[1]-1) / (min(X.shape[0], X.shape[1]-1) - 2)
            else:
                self.m = 2
        self.n_dim = X.shape[1]
        self.v0 = np.mean(X, axis=0, keepdims=True)  # v0: data center
        self.weight_ = np.ones([self.n_cluster, self.n_dim])/self.n_dim
        if self.init_method == 'random':
            self.center_ = X[np.random.choice(np.arange(X.shape[0]), replace=False, size=self.n_cluster), :]  # init by k-mean
        elif self.init_method == 'kmean':
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.n_cluster)
            km.fit(X)
            self.center_ = km.cluster_centers_
        else:
            raise ValueError('init method only supports [random, kmean]')
        # self.visual(X)

        loss = []

        old_V = None

        for i in range(self.tol_iter):
            self.U = self.update_u(X, self.weight_, self.center_, self.v0)  # U: [n_clusters, n_samples]

            self.center_ = self.update_v(X, self.U, self.v0)
            self.weight_ = self.update_w(X, self.U, self.center_, self.v0)

            loss.append(self.overall_loss(X))
            if i >= 1 and math.sqrt(np.sum((self.center_ - old_V)**2)) < self.error:
                break
            else:
                old_V = self.center_.copy()
        self.loss = loss
        self.fitted = True

        self.variance_ = np.zeros(self.center_.shape)  # n_clusters * n_dim

        for i in range(self.n_dim):
            self.variance_[:, i] = np.sum(
                self.U * ((X[:, i][:, np.newaxis] - self.center_[:, i].T) ** 2).T, axis=1
            ) / np.sum(self.U, axis=1)
        self.variance_ *= self.scale
        self.variance_ = np.fmax(self.variance_, np.finfo(np.float64).eps)
        self.norm_sparse_thres = np.percentile(self.weight_.reshape([-1]), self.sparse_thres)

        return self

    def update_u(self, X, W, V, v0):
        v = np.expand_dims(V.T, axis=0)
        x = np.expand_dims(X, axis=2)
        d1 = np.sum((x-v)**2 * W.T, axis=1)
        d2 = np.sum((V-v0)**2 * W, axis=1)[np.newaxis, :]
        d2 = np.repeat(d2, repeats=d1.shape[0], axis=0)
        d2 = np.fmax(d2, np.finfo(np.float64).eps)
        min_eta = np.min(d1/d2, axis=1, keepdims=True)
        self.sample_eta = np.minimum(self.eta, min_eta)
        d = d1 - d2*self.sample_eta
        d = np.fmax(d, np.finfo(np.float64).eps)
        d = d ** (1/(1-self.m))
        d = d / np.sum(d, axis=1, keepdims=True)
        return d.T

    def update_v(self, X, U, v0):
        u = np.expand_dims(U**self.m, axis=2)
        wd = u * (X - self.sample_eta * v0)[np.newaxis, :, :]  # weighted distance
        wd = np.sum(wd, axis=1) / np.sum(u*(1-self.sample_eta), axis=1)
        return wd

    def update_w(self, X, U, V, v0):
        v = np.expand_dims(V.T, axis=0)
        x = np.expand_dims(X, axis=2)
        u = np.expand_dims(U.T, axis=1)
        d1 = np.sum((u**self.m) * ((x-v)**2), axis=0).T
        u = np.expand_dims(U.T, axis=2)
        d2 = np.sum((V-v0)**2 * (u**self.m), axis=0)
        sig = d1 - self.eta * d2
        sig = -sig/self.gamma
        sig = np.fmin(sig, 700)  # avoid overflow
        # print(np.max(sig))
        sig = np.exp(sig)
        sig = np.fmax(sig, np.finfo(np.float64).eps)  # avoid underflow
        sig = np.fmin(sig, np.finfo(np.float64).max)
        return sig / np.sum(sig, axis=1, keepdims=True)

    def overall_loss(self, X):
        v = np.expand_dims(self.center_.T, axis=0)
        x = np.expand_dims(X, axis=2)
        d1 = np.sum(np.sum((x-v)**2 * self.weight_.T, axis=1) * (self.U.T**self.m))
        return d1

    def __str__(self):
        return "ESSC"
