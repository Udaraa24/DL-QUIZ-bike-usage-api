from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import joblib
import torch

from app.model_def import MLPRegressor

app = FastAPI(title="Bike Demand Predictor", version="1.0.0")

# Load artifacts
ART = torch.load("model/model.pt", map_location="cpu")
PREPROCESS = joblib.load("model/preprocess.pkl")

MODEL = MLPRegressor(in_features=ART["input_dim"])
MODEL.load_state_dict(ART["model_state_dict"])
MODEL.eval()

class PredictRequest(BaseModel):
    hr: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    temp: float = Field(..., ge=0.0, le=1.0, description="Normalized temperature (0-1) from dataset format")
    hum: float = Field(..., ge=0.0, le=1.0, description="Normalized humidity (0-1)")
    windspeed: float = Field(..., ge=0.0, le=1.0, description="Normalized windspeed (0-1)")
    workingday: int = Field(..., ge=0, le=1, description="1 if working day else 0")
    season: int = Field(..., ge=1, le=4, description="1=spring,2=summer,3=fall,4=winter")
    weathersit: int = Field(..., ge=1, le=4, description="1..4 (clear->heavy rain/snow)")

class PredictResponse(BaseModel):
    predicted_cnt: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Build a single-row “dataframe-like” dict for preprocess
    row = {
        "hr": [req.hr],
        "temp": [req.temp],
        "hum": [req.hum],
        "windspeed": [req.windspeed],
        "workingday": [req.workingday],
        "season": [req.season],
        "weathersit": [req.weathersit],
    }

    import pandas as pd
    X = pd.DataFrame(row)
    Xp = PREPROCESS.transform(X)
    Xp = Xp.toarray().astype(np.float32) if hasattr(Xp, "toarray") else Xp.astype(np.float32)

    with torch.no_grad():
        pred = MODEL(torch.from_numpy(Xp)).item()

    # Rental counts can't be negative
    pred = max(0.0, float(pred))
    return PredictResponse(predicted_cnt=pred)
