import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Use a simple relative path for the tracking URI.
mlflow.set_tracking_uri("file:./mlruns")
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

        # Log the model and its signature
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=mlflow.models.infer_signature(input_example, model.predict(input_example)),
            input_example=input_example
        )
        return model_info, accuracy

# --- Train Models ---
lr_params = {'C': 1.0, 'solver': 'liblinear'}
lr_model_info, lr_accuracy = train_model(LogisticRegression(**lr_params), "LogisticRegression", lr_params)

rf_params = {'n_estimators': 100, 'max_depth': 5}
rf_model_info, rf_accuracy = train_model(RandomForestClassifier(**rf_params), "RandomForestClassifier", rf_params)

# --- Automatically Promote the Best Model ---
model_name = "IrisClassifier"
client = mlflow.tracking.MlflowClient()

if lr_accuracy >= rf_accuracy:
    best_model_uri = lr_model_info.model_uri
    print(f"Logistic Regression is the best model with accuracy: {lr_accuracy}. Registering it.")
else:
    best_model_uri = rf_model_info.model_uri
    print(f"Random Forest is the best model with accuracy: {rf_accuracy}. Registering it.")

# Register the best model
registered_model = mlflow.register_model(best_model_uri, model_name)

# Set the "prod" alias on the newly registered model version
client.set_registered_model_alias(
    name=model_name,
    alias="prod",
    version=registered_model.version
)
print(f"Model '{model_name}' version '{registered_model.version}' has been registered and assigned the 'prod' alias.")
