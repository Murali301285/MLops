import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import subprocess
import os
from prometheus_client import Counter, make_asgi_app

# --- Configuration & Setup ---


log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, "predictions.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Log to a file
        logging.StreamHandler()         # Log to the console
    ]
)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI(title="Iris Classifier API")
model = None

# --- Prometheus Metrics ---
prediction_counter = Counter("prediction_requests_total", "Total number of prediction requests")
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.on_event("startup")
def load_model():
    global model
    model_path = "./model_artifacts/model" 

    try:
        model = mlflow.pyfunc.load_model(model_uri=model_path)
        logging.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}. Error: {e}")
        model = None

@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please check server logs.")

    try:
        # Increment Prometheus counter
        prediction_counter.inc()

        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)
        prediction_value = int(prediction[0])
        
        # Log request and response to file (and console)
        logging.info(f"INPUT: {data.dict()} -> PREDICTION: {prediction_value}")
        
        return {"prediction": prediction_value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/retrain", status_code=202)
def retrain():
    logging.info("Retraining trigger received.")
    try:
        # Run the training script as a subprocess
      
        
        result = subprocess.run(
            ["python", "scripts/train.py"],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"Retraining script finished successfully.\n{result.stdout}")
        
        # Reload the model after retraining
        load_model()
        
        return {"status": "Retraining successful", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        logging.error(f"Retraining script failed.\n{e.stderr}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e.stderr}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during retraining: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API. Use /docs to see the API documentation."}