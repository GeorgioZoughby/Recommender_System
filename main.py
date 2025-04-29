# MAIN ENTRY OF THE PROGRAM WILL GO HERE

from recommender import RecommenderSystem
import pandas as pd
import numpy as np
if __name__ == "__main__":
    data = {
        'HP1': [4, 5, 2, 3],
        'HP2': [5, 5, 4, 3],
        'HP3': [1, 4, 5, np.nan],
        'TW': [np.nan, np.nan, np. nan, np.nan],
        'SW1': [np.nan, np.nan, np.nan, np.nan],
        'SW2': [np.nan, np.nan, np.nan, np.nan],
        'SW3': [np.nan, np.nan, np.nan, np.nan],
    }
    df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])

    recommender = RecommenderSystem()
    recommender.fit(df)

    for method in ["user", "item", "baseline"]:
        preds = recommender.predict(method=method)
        recommender.pretty_print_matrix(preds)
        recommender.plot_heatmap(preds, title=f"Predicted Ratings ({method.capitalize()} CF)")
