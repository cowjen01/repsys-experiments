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


# class BookCrossing(Dataset):
#     def __init__(self) -> None:
#         self.books = pd.read_csv("./data/bx-csv/books.csv", sep=";", escapechar="\", encoding="CP1252")

#     def name(self):
#         return "bx"

#     def item_cols(self):
#         return {
#             "ISBN": dtypes.ItemID(),
#             "Book-Title": dtypes.Title(),
#             "Book-Author": dtypes.Category(),
#             "Year-Of-Publication": dtypes.Number(data_type=int),
#             # "Publisher": dtypes.Category(),
#             "Image-URL-L": dtypes.String()
#         }

#     def interaction_cols(self):
#         return {
#             "ISBN": dtypes.ItemID(),
#             "User-ID": dtypes.UserID(),
#         }

#     def load_items(self):
#         return self.books

#     def load_interactions(self):
#         df = pd.read_csv("./data/bx-csv/ratings.csv", sep=";", escapechar="\", encoding="CP1252")

#         df = df[df["ISBN"].isin(self.books["ISBN"].unique())]
#         df = df[df["Book-Rating"] == 0]

#         return df
