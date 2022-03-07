import os
import pickle
from abc import ABC

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

    def _apply_filters(self, predictions, **kwargs):
        if kwargs.get("genre"):
            items = self.dataset.items
            items = items[items["genre"].apply(lambda x: kwargs.get("genre") not in x)]
            indices = items.index.map(self.dataset.item_id_to_index)
            predictions[:, indices] = 0

    def web_params(self):
        return {
            "genre": Select(options=self.dataset.tags.get("genre")),
        }
