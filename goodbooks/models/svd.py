import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd

from models.base import BaseModel


class SVD(BaseModel):
    def __init__(self, n=30):
        self.sim = None
        self.n = n

    def name(self) -> str:
        return f"svd{self.n}"

    def _save_model(self):
        self._create_checkpoints_dir()
        np.save(self._checkpoint_path(), self.sim)

    def _load_model(self):
        self.sim = np.load(self._checkpoint_path())

    def fit(self, training=False):
        if training:
            X = self.dataset.get_train_data()
            U, sigma, VT = randomized_svd(
                X, n_components=self.n, n_iter=10, random_state=self.config.seed
            )
            self.sim = VT.T.dot(VT)
            self._save_model()
        else:
            self._load_model()

    def predict(self, X: csr_matrix, **kwargs):
        X_predict = X.dot(self.sim)
        X_predict[X.nonzero()] = 0

        self._apply_filters(X_predict, **kwargs)

        return X_predict

class SVD1(SVD):
    def __init__(self):
        super().__init__(1)

class SVD5(SVD):
    def __init__(self):
        super().__init__(5)

class SVD50(SVD):
    def __init__(self):
        super().__init__(50)

class SVD500(SVD):
    def __init__(self):
        super().__init__(500)