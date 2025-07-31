import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI(title="Iris Classifier API")
model = None

@app.on_event("startup")
def load_model():
    global model
    model_name = "IrisClassifier"
    try:
        # Load the model with the 'prod' alias
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@prod")
        print(f"Successfully loaded model '{model_name}@prod'")
    except mlflow.exceptions.MlflowException:
        print(f"Could not find model with alias 'prod'. Falling back to latest version.")
        try:
            # Fallback to the latest version if 'prod' alias doesn't exist
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(name=model_name)[0].version
            model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{latest_version}")
            print(f"Successfully loaded latest model version: {latest_version}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None

@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please check server logs.")

    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API. Use /docs to see the API documentation."}