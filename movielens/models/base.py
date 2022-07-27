import os
from abc import ABC

import numpy as np
from repsys import Model
from repsys.ui import Select


class BaseModel(Model, ABC):
    def _checkpoint_path(self):
        return os.path.join("./checkpoints", f"{self.name()}.npy")

    def _create_checkpoints_dir(self):
        dir_path = os.path.dirname(self._checkpoint_path())
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def _mask_items(self, X_predict, item_indices):
        mask = np.ones(self.dataset.items.shape[0], dtype=np.bool)
        mask[item_indices] = 0
        X_predict[:, mask] = 0

    def _filter_items(self, X_predict, col, value):
        if value:
            indices = self.dataset.filter_items_by_tags(col, [value])
            self._mask_items(X_predict, indices)

    def _apply_filters(self, X_predict, **kwargs):
        self._filter_items(X_predict, "genre", kwargs.get("genre"))

    def web_params(self):
        return {"genre": Select(options=self.dataset.tags.get("genre"))}
