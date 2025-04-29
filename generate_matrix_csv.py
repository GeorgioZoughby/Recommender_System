import pandas as pd
import numpy as np

# Settings
num_users = 10000
num_items = 25

# Generate random ratings with 70% missing (sparse)
ratings = np.random.choice([1, 2, 3, 4, 5, None], size=(num_users, num_items), p=[0.06, 0.06, 0.06, 0.06, 0.06, 0.7])
df = pd.DataFrame(ratings, columns=[f"Item_{i+1}" for i in range(num_items)])
df.insert(0, "user", [f"user_{i+1}" for i in range(num_users)])

# Save the CSV
df.to_csv("ratings_matrix_10000.csv", index=False)