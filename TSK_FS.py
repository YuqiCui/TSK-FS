import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from lib.fuzzy_cluster import ESSC
from scipy.special import softmax
import copy

class FuzzyBoostClassifier:
    def __init__(self, estimator='logistic', estimator_c=1., n_cluster=2, cluster_m=2, cluster_eta=0.1,
                 cluster_gamma=0.1, cluster_scale=1., order=1):
        """

        :param estimator: Classifier type, 'logistic' for LogisticRegression, 'ridge' for RidgeClassifier
        :param estimator_c: L2 regularization for estimator, 'C' in LogisticRegression, 'alpha' in RidgeClassifier
        :param n_cluster: number of clusters
        :param cluster_m: fuzzy index $m$ for ESSC
        :param cluster_eta: parameter $\eta$ for ESSC
        :param cluster_gamma: parameter $\gamma$ for ESSC
        :param cluster_scale: scale parameter for ESSC
        :param order: 0 or 1, order of TSK
        """
        self.estimator = estimator
        self.estimator_c = estimator_c
        self.n_cluster = n_cluster
        self.cluster_m = cluster_m
        self.cluster_eta = cluster_eta
        self.cluster_gamma = cluster_gamma
        self.cluster_scale = cluster_scale
        self.order = order


    def get_params(self, deep=True):
        return {
            'estimator': self.estimator,
            'estimator_c': self.estimator_c,
            'n_cluster': self.n_cluster,
            'cluster_m': self.cluster_m,
            'cluster_eta': self.cluster_eta,
            'cluster_gamma': self.cluster_gamma,
            'cluster_scale': self.cluster_scale,
            'order': self.order,
        }

    def set_params(self, **params):
        for p, v in params.items():
            setattr(self, p, v)
        return self


    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.center_, self.delta_ = self.__cluster__(X, self.n_cluster, self.cluster_eta, self.cluster_gamma, self.cluster_scale)  # [n_cluster, n_features], [n_cluster, n_features]
        self.curr_n_cluster = self.n_cluster
        self.experts_ = self.__fit_experts_by_firing_level__(X, y, self.__firing_level__(X, self.center_, self.delta_))

        self.no_boost = copy.deepcopy(self)
        return self

    def predict(self, X, y=None):
        prob = self.predict_proba(X)
        if self.n_classes == 2:
            return np.where(prob > 0.5, 1, 0)
        else:
            return np.argmax(prob, axis=1)

    def predict_proba(self, X, center=None, delta=None, experts=None):
        if center is None:
            center = self.center_
        if delta is None:
            delta = self.delta_
        if experts is None:
            experts = self.experts_
        mem = self.__firing_level__(X, center, delta)
        xp = self.x2xp(X, mem, self.order)
        out = experts.decision_function(xp)
        if self.n_classes == 2:
            return self.__logits_to_prob_2_class__(out)
        else:
            return self.__logits_to_prob_multi_class__(out)

    @staticmethod
    def __logits_to_prob_2_class__(logits):
        """
        Converting 2 class raw score to probability
        :param logits:
        :return:
        """
        return 1/(1+np.exp(-logits))

    @staticmethod
    def __logits_to_prob_multi_class__(logits, axis=1):
        """
        Converting multi-class raw score to probability
        :param logits:
        :return:
        """
        return softmax(logits, axis=axis)

    @staticmethod
    def __cluster__(data, n_cluster, eta, gamma, scale):
        """
        Comute data centers and membership of each point by ESSC, and compute the variance of each feature
        :param data: n_Samples * n_Features
        :param n_cluster: number of center
        :return: centers: data center, delta: variance of each feature
        """
        fuzzy_cluster = ESSC(n_cluster, eta=eta, gamma=gamma, tol_iter=100, scale=scale).fit(data)
        centers = fuzzy_cluster.center_
        delta = fuzzy_cluster.variance_
        return centers, delta

    @staticmethod
    def __firing_level__(data, centers, delta):
        """
        Compute firing strength using Gaussian model
        :param data: n_Samples * n_Features
        :param centers: data center，n_Clusters * n_Features
        :param delta: variance of each feature， n_Clusters * n_Features
        :return: firing strength
        """
        d = -(np.expand_dims(data, axis=2) - np.expand_dims(centers.T, axis=0))**2 / (2 * delta.T)
        d = np.exp(np.sum(d, axis=1))
        d = np.fmax(d, np.finfo(np.float64).eps)
        return d / np.sum(d, axis=1, keepdims=True)

    @staticmethod
    def x2xp(X, mem, order=1):
        """
        Converting raw input feature X to TSK consequent input
        :param X: raw input, [n_sample, n_features]
        :param mem: firing level of each rule, [n_sample, n_clusters]
        :param order:order of TSK, 0 or 1
        :return:
        """
        if order == 0:
            return mem
        else:
            N = X.shape[0]
            mem = np.expand_dims(mem, axis=1)
            X = np.expand_dims(np.concatenate((X, np.ones([N, 1])), axis=1), axis=2)
            X = np.repeat(X, repeats=mem.shape[1], axis=2)
            xp = X * mem
            xp = xp.reshape([N, -1])
            return xp

    def __fit_experts_by_firing_level__(self, X, y, mu_a):
        """

        :param X:
        :param y:
        :param mu_a: firing level, [n_sample, n_clusters]
        :return:
        """
        if self.estimator == 'logistic':
            experts = LogisticRegression(C=self.estimator_c, max_iter=200)
        elif self.estimator == 'ridge':
            experts = RidgeClassifier(self.estimator_c)
        else:
            raise ValueError("Estimator type only support logistic and ridge, not {}".format(self.estimator))

        xp = self.x2xp(X, mu_a, self.order)
        experts.fit(xp, y)
        return experts

