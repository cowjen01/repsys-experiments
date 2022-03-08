import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

import repsys.dtypes as dtypes
from repsys.dataset import Dataset


class MovieLens(Dataset):
    def __init__(self):
        self.nmf = NMF(n_components=15, init="nndsvd", verbose=2)

    def name(self):
        return "ml20m"

    def compute_embeddings(self, X: csr_matrix):
        W = self.nmf.fit_transform(X)
        H = self.nmf.components_
        return W, H.T

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
        return {
            "movieId": dtypes.ItemID(),
            "userId": dtypes.UserID(),
            "rating": dtypes.Interaction(),
        }

    def load_items(self):
        df = pd.read_json("./data/ml-20m/items.json")
        df["year"] = df["product_name"].str.extract(r"\((\d+)\)")
        return df

    def load_interactions(self):
        df = pd.read_csv("./data/ml-20m/ratings.csv")
        df = df[df["rating"] > 3.5]
        df["rating"] = 1
        return df


# class GoodBooks(Dataset):
#     def name(self):
#         return "gb"

#     def item_cols(self):
#         return {
#             "book_id": dtypes.ItemID(),
#             "isbn": dtypes.String(),
#             "authors": dtypes.String(),
#             "year": dtypes.Number(data_type=int),
#             "original_title": dtypes.Title(),
#             "language_code": dtypes.Category(),
#             "average_rating": dtypes.Number(),
#             "image_url": dtypes.String()
#         }

#     def interaction_cols(self):
#         return {
#             "book_id": dtypes.ItemID(),
#             "user_id": dtypes.UserID(),
#         }

#     def load_items(self):
#         return pd.read_csv("./datasets/gb/books.csv")

#     def load_interactions(self):
#         df = pd.read_csv("./datasets/gb/ratings.csv")
#         df = df[df['rating'] > 3]
#         return df
