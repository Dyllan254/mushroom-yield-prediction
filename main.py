import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import logging
from fastapi.responses import JSONResponse
import os
import uvicorn
import io

 # Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and scaler
model = joblib.load("stacking_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input schema
class MushroomInput(BaseModel):
    temperature: float
    humidity: float
    ph: float
    co2: float

# JSON-based prediction endpoint
@app.post("/predict")
def predict(data: MushroomInput):
    try:
        logger.info("Received JSON prediction request: %s", data)
        input_data = np.array([[data.temperature, data.humidity, data.ph, data.co2]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        result = {"predicted_yield_kg": round(float(prediction[0]), 2)}
        logger.info("Prediction result: %s", result)
        return result
    except Exception as e:
        logger.error("Error during JSON prediction: %s", str(e))
        return JSONResponse(status_code=500, content={"error": "Internal server error during prediction."})

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Mushroom Yield Prediction"}