import pandas as pd

import repsys.dtypes as dtypes
from repsys.dataset import Dataset

class GoodBooks(Dataset):
    def name(self):
        return "gb"

    def item_cols(self):
        return {
            "book_id": dtypes.ItemID(),
            "original_title": dtypes.Title(),
            "storyline": dtypes.String(),
            "image_url_x": dtypes.String(),
            "authors": dtypes.String(),
            "cat3": dtypes.Tag(sep="|"),
            "year": dtypes.Number(data_type=int),
        }

    def interaction_cols(self):
        return {
            "book_id": dtypes.ItemID(),
            "user_id": dtypes.UserID()
        }

    def load_items(self):
        df = pd.read_csv("./data/gb/items.csv")
        df["year"] = df["original_publication_year"].fillna(0).astype(int)
        return df

    def load_interactions(self):
        df = pd.read_csv("./data/gb/ratings.csv")
        df = df[df["rating"] >= 4.]
        return df

    def web_default_config(self):
        return {
            "mappings": {"title":"original_title","subtitle":"authors","caption":"year","image":"image_url_x","content":"storyline"},
            "recommenders": [{"name":"KNN","itemsPerPage":4,"itemsLimit":20,"model":"knn","modelParams":{}},{"name":"EASE","itemsPerPage":4,"itemsLimit":20,"model":"ease","modelParams":{}},{"name":"TopPop","itemsPerPage":4,"itemsLimit":20,"model":"pop","modelParams":{}},{"name":"SVD","itemsPerPage":4,"itemsLimit":20,"model":"svd","modelParams":{}}]
        }

