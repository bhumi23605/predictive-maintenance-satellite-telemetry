from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load models
model = joblib.load("bestmodel.pkl")


class InputData(BaseModel):
    features: list  # List of input features

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    pred = model.predict(features)[0]
    return {"prediction": int(pred)}
