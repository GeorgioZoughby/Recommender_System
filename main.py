from recommender import RecommenderSystem
import pandas as pd
import numpy as np

# MAIN ENTRY OF THE PROGRAM WILL GO HERE
if __name__ == "__main__":
    df = pd.read_csv("ratings_matrix_10000.csv", index_col=0)

    recommender = RecommenderSystem()
    recommender.fit(df)

    for method in ["user", "item", "baseline"]:
        preds = recommender.predict(method=method)
        recommender.pretty_print_matrix(preds)
        recommender.plot_heatmap(preds, title=f"{method.title()} CF Predictions", filename=f"{method}_cf_heatmap.png")
