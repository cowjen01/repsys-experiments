import os
import pickle
from abc import ABC
import numpy as np

from repsys import Model
from repsys.ui import Select


class BaseModel(Model, ABC):
    def _checkpoint_path(self):
        return os.path.join("./checkpoints", self.name())

    def _serialize(self):
        return self.model

    def _deserialize(self, checkpoint):
        self.model = checkpoint

    def _load_model(self):
        if not os.path.exists(self._checkpoint_path()):
            raise Exception("The model has not been trained yet.")

        checkpoint = pickle.load(open(self._checkpoint_path(), "rb"))
        self._deserialize(checkpoint)

    def _save_model(self):
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")

        checkpoint = open(self._checkpoint_path(), "wb")
        pickle.dump(self._serialize(), checkpoint)

    def _mask_items(self, X_predict, item_indices):
        mask = np.ones(self.dataset.items.shape[0], dtype=np.bool)
        mask[item_indices] = 0
        X_predict[:, mask] = 0

    def _apply_filters(self, X_predict, **kwargs):
        if kwargs.get("genre"):
            indices = self.dataset.filter_items_by_tag("genre", kwargs.get("genre"))
            X_predict[:, indices] = 0

    def web_params(self):
        return {
            "genre": Select(options=self.dataset.tags.get("genre")),
        }
