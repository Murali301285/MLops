# create_data.py
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save to the data directory
df.to_csv('data/iris.csv', index=False)
print("data/iris.csv created successfully.")