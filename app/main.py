import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # FIX: Point to the nested 'model' directory
    model_path = "./model_artifacts/model" # Path inside the container

    try:
        # Load the model from the local directory
        model = mlflow.pyfunc.load_model(model_uri=model_path)
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}. Error: {e}")
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