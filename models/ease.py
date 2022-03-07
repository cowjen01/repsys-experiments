# class EASE(BaseModel):
#     def __init__(self, l2_lambda=0.5):
#         self.B = None
#         self.l2_lambda = l2_lambda
#
#     def name(self) -> str:
#         return "ease"
#
#     def _serialize(self):
#         return self.B
#
#     def _deserialize(self, checkpoint):
#         self.B = checkpoint
#
#     def fit(self, training: bool = False) -> None:
#         if training:
#             X = self.dataset.get_train_data()
#             G = X.T.dot(X).toarray()
#
#             diagonal_indices = np.diag_indices(G.shape[0])
#             G[diagonal_indices] += self.l2_lambda
#
#             P = np.linalg.inv(G)
#             B = P / (-np.diag(P))
#             B[diagonal_indices] = 0
#
#             self.B = B
#             self._save_model()
#         else:
#             self._load_model()
#
#     def predict(self, X: csr_matrix, **kwargs):
#         predictions = X.dot(self.B)
#         predictions[X.nonzero()] = 0
#
#         self._apply_filters(predictions, **kwargs)
#
#         return predictions
