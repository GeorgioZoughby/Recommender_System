import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from termcolor import colored


class RecommenderSystem:
    def __init__(self):
        self.ratings = None
        self.global_mean = None

    def fit(self, df):
        self.ratings = df.copy()
        self.global_mean = np.nanmean(df.values)

    def _cosine_similarity(self, matrix):
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        norm_matrix = matrix / (norm + 1e-10)
        similarity = np.dot(norm_matrix, norm_matrix.T)
        return similarity

    def _user_user_cf(self):
        matrix = self.ratings.fillna(0).values
        similarity = self._cosine_similarity(matrix)
        predictions = np.zeros(matrix.shape)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if self.ratings.isna().iloc[i, j]:
                    sim_scores = similarity[i]
                    item_ratings = matrix[:, j]
                    mask = item_ratings != 0
                    if np.sum(mask) == 0:
                        pred = self.global_mean
                    else:
                        pred = np.dot(sim_scores[mask], item_ratings[mask]) / (np.sum(np.abs(sim_scores[mask])) + 1e-10)
                    predictions[i, j] = pred
                else:
                    predictions[i, j] = matrix[i, j]
        return predictions

    def _item_item_cf(self):
        matrix = self.ratings.fillna(0).values.T  # Transpose for item-item
        similarity = self._cosine_similarity(matrix)
        predictions = np.zeros(matrix.shape)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if self.ratings.isna().iloc[j, i]:
                    sim_scores = similarity[i]
                    user_ratings = matrix[:, j]
                    mask = user_ratings != 0
                    if np.sum(mask) == 0:
                        pred = self.global_mean
                    else:
                        pred = np.dot(sim_scores[mask], user_ratings[mask]) / (np.sum(np.abs(sim_scores[mask])) + 1e-10)
                    predictions[i, j] = pred
                else:
                    predictions[i, j] = matrix[i, j]
        return predictions.T

    def _global_baseline(self):
        user_means = self.ratings.mean(axis=1)
        item_means = self.ratings.mean(axis=0)

        predictions = np.zeros(self.ratings.shape)
        for i in range(self.ratings.shape[0]):
            for j in range(self.ratings.shape[1]):
                if pd.isna(self.ratings.iloc[i, j]):
                    pred = self.global_mean
                    if not np.isnan(user_means.iloc[i]):
                        pred += (user_means.iloc[i] - self.global_mean)
                    if not np.isnan(item_means.iloc[j]):
                        pred += (item_means.iloc[j] - self.global_mean)
                    predictions[i, j] = pred
                else:
                    predictions[i, j] = self.ratings.iloc[i, j]
        return predictions

    def predict(self, method="user"):
        if method == "user":
            print(colored("\n[Using User-User Collaborative Filtering]", "cyan"))
            return self._user_user_cf()
        elif method == "item":
            print(colored("\n[Using Item-Item Collaborative Filtering]", "cyan"))
            return self._item_item_cf()
        elif method == "baseline":
            print(colored("\n[Using Global Baseline Estimate]", "cyan"))
            return self._global_baseline()
        else:
            raise ValueError("Unknown method. Choose 'user', 'item', or 'baseline'.")

    def evaluate(self, predictions, true_ratings):
        mask = ~np.isnan(true_ratings.values)
        mse = mean_squared_error(true_ratings.values[mask], predictions[mask])
        rmse = np.sqrt(mse)
        print(colored(f"\n[Evaluation RMSE: {rmse:.4f}]", "green"))
        return rmse

    def plot_heatmap(self, matrix, title="Ratings Heatmap"):
        plt.figure(figsize=(10, 6))
        sns.heatmap(matrix, annot=True, fmt=".1f", cmap="coolwarm")
        plt.title(title)
        plt.show()

    def pretty_print_matrix(self, matrix):
        print("\nMatrix:")
        df = pd.DataFrame(matrix, index=self.ratings.index, columns=self.ratings.columns)
        print(colored(df.round(2).to_string(), "yellow"))
