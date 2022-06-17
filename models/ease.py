import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd

from models.base import BaseModel


class EASE(BaseModel):
    def __init__(self):
        self.sim = None

    def name(self) -> str:
        return "ease"

    def _save_model(self):
        self._create_checkpoints_dir()
        np.save(self._checkpoint_path(), self.B)

    def _load_model(self):
        self.B = np.load(self._checkpoint_path())

    def fit(self, training=False):
        lambda_=100
        if training:
            X = self.dataset.get_train_data()
            self.X = X
            G = X.T.dot(X).toarray()
            diagIndices = np.diag_indices(G.shape[0])
            G[diagIndices] += lambda_
            P = np.linalg.inv(G)
            B = P / (-np.diag(P))
            B[diagIndices] = 0

            self.B = B
            self._save_model()
        else:
            self._load_model()

    def predict(self, X: csr_matrix, **kwargs):
        X_predict = X.dot(self.B)
        X_predict[X.nonzero()] = 0

        self._apply_filters(X_predict, **kwargs)

        return X_predict
