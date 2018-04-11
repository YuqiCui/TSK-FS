
class TSK_FS():
    def __init__(self, n_cluster=20, C=0.1):
        """
        Takagi-Sugeno-Kang Fuzzy System by Yuqi Cui.
        Follow the style of sklearn
        2018/4/7
        :param n_cluster: number of rules /  number of clusters
        :param C: L2 regularization coefficient
        :param method: 'classification' or 'regression', if classification then target should have size of n_Sample * n_Classes
        """
        self.n_cluster = n_cluster
        self.lamda = C
        self.trained = False

    def fit(self, X_train, y_train):
        """
        train TSK
        :param X_train: n_Samples * n_Features
        :param y_train: n_Samples * n_Classes or n_Samples * 1 if regression
        :return:
        """
        n_samples, n_features = X_train.shape
        n_cluster = self.n_cluster
        assert (n_samples == len(y_train)), \
            'X_train and y_train samples num must be same'
        centers, delta = self.__fcm__(X_train, n_cluster)
        self.centers = centers
        self.delta = delta
        # compute x_g
        xg = self.__gaussian_feature__(X_train, centers, delta)
        # train by pinv
        xg1 = np.dot(xg.T, xg)
        pg = np.linalg.pinv(xg1 + self.lamda * np.eye(xg1.shape[0])).dot(xg.T).dot(y_train)
        # pg = pg.dot(y_train)
        self.pg = pg
        # print(pg)
        self.trained = True

    def predict(self, X_test):
        """
        predict by test data
        :param X_test: n_Samples * n_Features
        :return: n_Samples * n_Classes or n_Samples * 1 if regression
        """
        assert(self.trained), "Error when predict, use fit first!"
        xg_test = self.__gaussian_feature__(X_test, self.centers, self.delta)
        y_pred = xg_test.dot(self.pg)
        return y_pred

    def fcm(self, data, n_cluster):
        return self.__fcm__(data, n_cluster)

    def gaussian_feature(self, data, centers, delta):
        return self.__gaussian_feature__(data, centers, delta)

    def __fcm__(self, data, n_cluster):
        """
        Comute data centers and membership of each point by FCM, and compute the variance of each feature
        :param data: n_Samples * n_Features
        :param n_cluster: number of center
        :return: centers: data center, delta: variance of each feature
        """
        n_samples, n_features = data.shape
        centers, mem, _, _, _, _, _ = fuzz.cmeans(
            data.T, n_cluster, 2.0, error=1e-5, maxiter=200)

        # compute delta compute the variance of each feature
        delta = np.zeros([n_cluster, n_features])
        for i in range(n_cluster):
            d = (data - centers[i, :]) ** 2
            delta[i, :] = np.sum(d * mem[i, :].reshape(-1, 1),
                                 axis=0) / np.sum(mem[i, :])

        return centers, delta

    def __gaussian_feature__(self, data, centers, delta):
        """
        Compute firing strength using Gaussian model
        :param data: n_Samples * n_Features
        :param centers: data center，n_Clusters * n_Features
        :param delta: variance of each feature， n_Clusters * n_Features
        :return: data_fs data的firing strength, n_Samples * [n_Clusters * (n_Features+1)]
        """
        n_cluster = self.n_cluster
        n_samples = data.shape[0]
        # compute firing strength of each data, n_Samples * n_Clusters
        mu_a = np.zeros([n_samples, n_cluster])
        for i in range(n_cluster):
            tmp_k = 0 - np.sum((data - centers[i, :]) ** 2 /
                               delta[i, :], axis=1)
            mu_a[:, i] = np.exp(tmp_k)  # exp max 709
        # norm
        mu_a = mu_a / np.sum(mu_a, axis=1, keepdims=True)
        # print(np.count_nonzero(mu_a!=mu_a))
        data_1 = np.concatenate((data, np.ones([n_samples, 1])), axis=1)
        zt = []
        for i in range(n_cluster):
            zt.append(data_1 * mu_a[:, i].reshape(-1, 1))
        data_fs = np.concatenate(zt, axis=1)
        data_fs = np.where(data_fs != data_fs, 1e-5, data_fs)
        return data_fs
