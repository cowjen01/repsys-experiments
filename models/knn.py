import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from models.base import BaseModel


class KNN(BaseModel):
    def __init__(self):
        self.model = NearestNeighbors(algorithm="brute", n_neighbors=20, metric="cosine")

    def name(self):
        return "knn"

    def fit(self, training=False):
        if training:
            X = self.dataset.get_train_data()
            self.model.fit(X)
            self._save_model()
        else:
            self._load_model()

    def predict(self, X, **kwargs):
        if X.count_nonzero() == 0:
            return np.random.uniform(size=X.shape)

        distances, indices = self.model.kneighbors(X)

        n_distances = distances[:, 1:]
        n_indices = indices[:, 1:]

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

        # indices = self.dataset.filter_items_by_tags("genre", ["Adventure"])
        # indices = self.dataset.filter_items_by_number('year', (2000, 2020))
        # self._mask_items(X_predict, indices)

        self._apply_filters(X_predict, **kwargs)

        return X_predict
