import pandas as pd

import repsys.dtypes as dtypes
from repsys.dataset import Dataset


class MovieLens(Dataset):
    def name(self):
        return "ml20m"

    def item_cols(self):
        return {
            "itemid": dtypes.ItemID(),
            "product_name": dtypes.Title(),
            "description": dtypes.String(),
            "image": dtypes.String(),
            "director": dtypes.String(),
            "language": dtypes.Tag(sep=", "),
            "genre": dtypes.Tag(sep=", "),
            "country": dtypes.Tag(sep=", "),
            "year": dtypes.Number(data_type=int),
        }

    def interaction_cols(self):
        return {"movieId": dtypes.ItemID(), "userId": dtypes.UserID()}

    def load_items(self):
        df = pd.read_json("./data/items.json")
        df["year"] = df["product_name"].str.extract(r"\((\d+)\)")
        return df

    def load_interactions(self):
        df = pd.read_csv("./data/ratings.csv")
        df = df[df["rating"] > 3.5]
        return df

    def web_default_config(self):
        return {
            "mappings": {
                "title": "product_name",
                "subtitle": "director",
                "caption": "year",
                "image": "image",
                "content": "description",
            },
            "recommenders": [
                {
                    "name": "KNN",
                    "model": "knn",
                },
                {
                    "name": "EASE",
                    "model": "ease",
                },
                {
                    "name": "TopPop",
                    "model": "pop",
                },
                {
                    "name": "SVD",
                    "model": "svd",
                },
            ],
        }
