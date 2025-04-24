import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from train import scaler

#load the trained model
model = joblib.load('model/stacking_model.pkl')

#create an instance of fastapi
app = FastAPI()

class MushroomInput(BaseModel):
    temperature: float
    humidity: float
    ph: float
    co2: float

@app.post("/predict")
def predict(data: MushroomInput):
    input_data = np.array([[data.temperature, data.humidity, data.ph, data.co2]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return {"predicted_yield_kg": round(prediction[0], 2)}

@app.get("/")
def read_root():
    return("Welcome to Mushroom Yield Prediction")