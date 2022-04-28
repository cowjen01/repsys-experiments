from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd

from models.base import BaseModel


class PureSVD(BaseModel):
    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.model = None

    def name(self) -> str:
        return "svd"

    def fit(self, training=False):
        if training:
            X = self.dataset.get_train_data()
            U, sigma, VT = randomized_svd(X, self.n_factors, random_state=self.config.seed)
            self.model = VT.T.dot(VT)
            self._save_model()
        else:
            self._load_model()

    def predict(self, X: csr_matrix, **kwargs):
        X_predict = X.dot(self.model)
        X_predict[X.nonzero()] = 0

        self._apply_filters(X_predict, **kwargs)

        return X_predict
