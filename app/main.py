from fastapi import FastAPI, HTTPException
from app.schemas import PredictionInput
from app.predict import make_prediction


app = FastAPI()


@app.post("/predict")
def predict(data: PredictionInput) -> dict:
    """Receive validated input, generate prediction using the model, and return the result."""
    try:
        prediction = make_prediction(data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
