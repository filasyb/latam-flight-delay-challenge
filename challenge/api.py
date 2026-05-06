from pathlib import Path
from typing import List

import fastapi
import pandas as pd
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from challenge.model import DelayModel

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.csv"

app = fastapi.FastAPI()
model = DelayModel()
training_data = pd.read_csv(DATA_PATH)
features, target = model.preprocess(
    data=training_data,
    target_column=DelayModel.TARGET_COLUMN
)
model.fit(features=features, target=target)

VALID_AIRLINES = set(training_data["OPERA"].unique())
VALID_FLIGHT_TYPES = {"I", "N"}


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("OPERA")
    def validate_airline(cls, value: str) -> str:
        if value not in VALID_AIRLINES:
            raise ValueError("Invalid airline")

        return value

    @validator("TIPOVUELO")
    def validate_flight_type(cls, value: str) -> str:
        if value not in VALID_FLIGHT_TYPES:
            raise ValueError("Invalid flight type")

        return value

    @validator("MES")
    def validate_month(cls, value: int) -> int:
        if value < 1 or value > 12:
            raise ValueError("Invalid month")

        return value


class PredictRequest(BaseModel):
    flights: List[Flight]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: fastapi.Request,
    exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    )

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    data = pd.DataFrame([flight.dict() for flight in request.flights])
    features = model.preprocess(data=data)
    predictions = model.predict(features=features)

    return {
        "predict": predictions
    }
