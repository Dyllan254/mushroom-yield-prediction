import numpy as np
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import os

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
import os

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

