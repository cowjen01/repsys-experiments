import numpy as np
from scipy.sparse import csr_matrix

from models.base import BaseModel


class EASE(BaseModel):
    def __init__(self, lmb: int = 100):
        self.sim = None
        self.lmb = lmb

    def name(self) -> str:
        return "ease"

    def _save_model(self):
        self._create_checkpoints_dir()
        np.save(self._checkpoint_path(), self.sim)

    def _load_model(self):
        self.sim = np.load(self._checkpoint_path())

    def fit(self, training=False):
        if training:
            X = self.dataset.get_train_data()
            G = X.T.dot(X).toarray()
            diagonal_indices = np.diag_indices(G.shape[0])
            G[diagonal_indices] += self.lmb
            P = np.linalg.inv(G)
            B = P / (-np.diag(P))
            B[diagonal_indices] = 0
            self.sim = B
            self._save_model()
        else:
            self._load_model()

    def predict(self, X: csr_matrix, **kwargs):
        X_predict = X.dot(self.sim)
        X_predict[X.nonzero()] = 0

        self._apply_filters(X_predict, **kwargs)

        return X_predict
