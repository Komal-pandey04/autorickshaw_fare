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
