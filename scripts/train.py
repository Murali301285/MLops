import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# Set the MLflow tracking URI. This will create a 'mlruns' directory in the project root.
mlflow.set_tracking_uri("file:" + os.path.join(os.path.dirname(__file__), '..', 'mlruns'))
mlflow.set_experiment("Iris_Classification")

print("Loading data...")
df = pd.read_csv('data/iris.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_example = X_train.head(1)

def train_model(model, model_name, params):
    with mlflow.start_run(run_name=model_name) as run:
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Accuracy: {accuracy}")

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=mlflow.models.infer_signature(input_example, model.predict(input_example)),
            input_example=input_example
        )
        return run.info.run_id, accuracy

# --- Train Models ---
lr_params = {'C': 1.0, 'solver': 'liblinear'}
lr_run_id, lr_accuracy = train_model(LogisticRegression(**lr_params), "LogisticRegression", lr_params)

rf_params = {'n_estimators': 100, 'max_depth': 5}
rf_run_id, rf_accuracy = train_model(RandomForestClassifier(**rf_params), "RandomForestClassifier", rf_params)

# --- Register the Best Model ---
best_run_id = lr_run_id if lr_accuracy > rf_accuracy else rf_run_id
model_uri = f"runs:/{best_run_id}/model"
registered_model = mlflow.register_model(model_uri, "IrisClassifier")
print(f"Model '{registered_model.name}' version '{registered_model.version}' registered.")