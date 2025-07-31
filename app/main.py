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
    model_name = "IrisClassifier"
    model_uri = f"models:/{model_name}@prod"
    print (model_uri)
    try:
        # Load the model with the 'prod' alias
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        logger.info(f"Successfully loaded model from {model_uri}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_uri}. Error: {e}")
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
