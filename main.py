from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("gru_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

app = FastAPI(
    title="GRU Time Series Forecast API",
    description="Predict next week's sales using a trained GRU model",
    version="1.0"
)
class ForecastRequest(BaseModel):
    sequence: list[float]  

@app.post("/predict")
def predict_next(request: ForecastRequest):
    sequence = request.sequence

    if len(sequence) != 12:
        raise HTTPException(status_code=400, detail="Sequence must contain exactly 12 values.")

    try:
        input_seq = np.array(sequence).reshape(-1, 1)  
        input_scaled = scaler.transform(input_seq)
        input_scaled = input_scaled.reshape(1, 12, 1)

        pred_scaled = model.predict(input_scaled)
        pred = scaler.inverse_transform(pred_scaled)

        return {
            "input_sequence": sequence,
            "predicted_weekly_sales": float(pred[0, 0])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
