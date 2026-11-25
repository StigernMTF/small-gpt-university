from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import Predictor

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["predictor"] = Predictor()
        print("Model loaded")
    except Exception as e:
        print(e)
        state["predictor"] = None
    yield

    state.clear()

app = FastAPI(title="My first ML project", lifespan=lifespan)

class PredictionResponce(BaseModel):
    label: str
    confidence: float
    probability: dict[str, float]

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100, examples=["this movie was awesome"])

@app.get("/health")
def health():
    return {"model_loaded": state.get("predictor") is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor.predict(request.text)