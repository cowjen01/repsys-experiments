import pandas as pd

import repsys.dtypes as dtypes
from repsys.dataset import Dataset


class MIND(Dataset):
    def name(self):
        return "mind"

    def item_cols(self):
        return {
            "itemid": dtypes.ItemID(),
            "title": dtypes.Title(),
            "abstract": dtypes.String(),
            "category": dtypes.Category(),
            "subcategory": dtypes.Category(),
        }

    def interaction_cols(self):
        return {"news_id": dtypes.ItemID(), "user_id": dtypes.UserID()}

    def load_items(self):
        return pd.read_csv("./data/mind/news_p10.csv", index_col=0)

    def load_interactions(self):
        return pd.read_csv("./data/mind/mind_interactions_train_p10.csv", index_col=0)

    def web_default_config(self):
        return {
            "mappings": {
                "title": "title",
                "subtitle": "subcategory",
                "caption": "category",
                "image": "",
                "content": "abstract",
            },
            "recommenders": [
                {"name": "KNN", "model": "knn"},
                {"name": "EASE", "model": "ease"},
                {"name": "TopPop", "model": "pop"},
                {"name": "SVD", "model": "svd"},
            ],
        }
