import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from models.base import BaseModel


class KNN(BaseModel):
    def __init__(self, k=50):
        self.model = None
        self.k = k

    def name(self):
        return f"knn{self.k}"

    def fit(self, training=False):
        X = self.dataset.get_train_data()
        self.model = NearestNeighbors(n_neighbors=self.k, metric="cosine")
        self.model.fit(X)

    def predict(self, X, **kwargs):
        if X.count_nonzero() == 0:
            return np.random.uniform(size=X.shape)

        n_distances, n_indices = self.model.kneighbors(X)

        n_distances = 1 - n_distances

        sums = n_distances.sum(axis=1)
        n_distances = n_distances / sums[:, np.newaxis]

        def f(dist, idx):
            A = self.dataset.get_train_data()[idx]
            D = sp.diags(dist)
            return D.dot(A).sum(axis=0)

        vf = np.vectorize(f, signature="(n),(n)->(m)")
        X_predict = vf(n_distances, n_indices)

        X_predict[X.nonzero()] = 0

        self._apply_filters(X_predict, **kwargs)

        return X_predict


class KNN1(KNN):
    def __init__(self):
        super().__init__(1)


class KNN5(KNN):
    def __init__(self):
        super().__init__(5)


class KNN500(KNN):
    def __init__(self):
        super().__init__(500)


class KNN5000(KNN):
    def __init__(self):
        super().__init__(5000)
