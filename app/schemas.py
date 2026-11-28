"""Request/response schemas. The client's two rules live here so bad input never reaches the model."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")   # exactly these fields, nothing else

    distance_km: float = Field(..., gt=0, le=40, description="Ride distance; above 40 km is impossible")
    wait_min: float = Field(..., ge=0, description="Waiting time in minutes; negative is a typing mistake")
    passengers: int = Field(..., ge=1, le=8)
    hour: int = Field(..., ge=0, le=23)
    luggage_bags: int = Field(..., ge=0, le=10)
    fuel_price_rs: float = Field(..., gt=0, le=500)
    booking: Literal["street", "app", "stand"]
    zone: Literal["Station Area", "Market", "Outskirts"]


class PredictResponse(BaseModel):
    prediction: float = Field(..., description="Predicted fare in rupees")


class BatchRequest(BaseModel):
    """Many rides at once (e.g. a day's bookings). Each ride follows the same two client rules."""
    model_config = ConfigDict(extra="forbid")
    rides: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: list[float]
    count: int


class ExplainResponse(BaseModel):
    prediction: float
    baseline_fare: float = Field(..., description="Fare of the typical ride in params.yaml")
    effects_rs: dict[str, float] = Field(..., description="Rs each input adds (+) or removes (-) vs the typical ride")
    other_effects_rs: float = Field(..., description="Interaction remainder so the numbers reconcile")
