import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target


df.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in df.columns]

# Save to the data directory
df.to_csv('data/iris.csv', index=False)
print("data/iris.csv created successfully with clean column names.")