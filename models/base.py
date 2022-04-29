import os
from abc import ABC

import numpy as np
from repsys import Model
from repsys.ui import Select


class BaseModel(Model, ABC):
    def _checkpoint_path(self):
        return os.path.join("./checkpoints", self.dataset.name(), f"{self.name()}.npy")

    def _create_checkpoints_dir(self):
        dir_path = os.path.dirname(self._checkpoint_path())
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    def _mask_items(self, X_predict, item_indices):
        mask = np.ones(self.dataset.items.shape[0], dtype=np.bool)
        mask[item_indices] = 0
        X_predict[:, mask] = 0

    def _apply_filters(self, X_predict, **kwargs):
        if kwargs.get("genre"):
            selected_genre = kwargs.get("genre")
            indices = self.dataset.filter_items_by_tags("genre", [selected_genre])
            self._mask_items(X_predict, indices)

    def web_params(self):
        if self.dataset.name() == "ml20m":
            return {"genre": Select(options=self.dataset.tags.get("genre"))}
        else:
            return {}
