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
        df = pd.read_json("./data/ml-20m/items.json")
        df["year"] = df["product_name"].str.extract(r"\((\d+)\)")
        return df

    def load_interactions(self):
        df = pd.read_csv("./data/ml-20m/ratings.csv")
        df = df[df["rating"] > 3.5]
        return df

    def default_web_config(self):
        return {
            "mappings": {"title":"product_name","subtitle":"director","caption":"year","image":"image","content":"description"},
            "recommenders": [{"name":"KNN","itemsPerPage":4,"itemsLimit":20,"model":"knn","modelParams":{"genre":""}},{"name":"SVD","itemsPerPage":4,"itemsLimit":20,"model":"svd","modelParams":{"genre":""}},{"name":"TopPop","itemsPerPage":4,"itemsLimit":20,"model":"pop","modelParams":{"genre":""}}]
        }


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
            # "language": dtypes.Tag(sep=", "),
            "cat3": dtypes.Tag(sep="|"),
            # "country": dtypes.Tag(sep=", "),
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

    def default_web_config(self):
        return {
            "mappings": {"title":"original_title","subtitle":"authors","caption":"year","image":"image_url_x","content":"storyline"},
            "recommenders": [{"name":"KNN","itemsPerPage":4,"itemsLimit":20,"model":"knn"},{"name":"SVD","itemsPerPage":4,"itemsLimit":20,"model":"svd"},{"name":"TopPop","itemsPerPage":4,"itemsLimit":20,"model":"pop"}]
        }

