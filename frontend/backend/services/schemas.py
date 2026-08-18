"""
Internal, provider-agnostic data models.

The rest of the application (feature engineering, predictor, API routes)
depends ONLY on these models -- never on the raw CricketData/CricAPI JSON
shape directly. If we ever swap data providers, only services/cricket_api.py
needs to change.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class MatchInfo(BaseModel):
    """Static/summary info about a match, used for the /matches list."""

    match_id: str
    name: str
    status: str
    venue: Optional[str] = None
    date: Optional[str] = None
    team1: Optional[str] = None
    team2: Optional[str] = None
    match_type: Optional[str] = None  # t20 / odi / test
    is_live: bool = False


class BallState(BaseModel):
    """A single normalized delivery, used for optional ball-history features."""

    over: float
    runs: int
    is_wicket: bool = False
    is_boundary: bool = False


class MatchState(BaseModel):
    """
    Normalized second-innings match state, sufficient to run the phase-1
    win-probability model (see services/feature_engineering.py).

    All fields are required for a prediction to be produced. If the match
    has not reached the second innings yet (no target set), predictions are
    not attempted -- see predictor.py.
    """

    match_id: str
    batting_team: str
    bowling_team: str
    venue: Optional[str] = None
    toss_winner: Optional[str] = None
    toss_decision: Optional[str] = None

    runs_so_far: int
    wickets_fallen: int = Field(ge=0, le=10)
    balls_bowled: int = Field(ge=0, le=120)

    target: Optional[int] = None  # None => first innings, no prediction possible

    innings: int = Field(ge=1, le=2)
    status: str = "live"


class PredictionInput(BaseModel):
    """The exact, ordered feature vector the model consumes."""

    features: dict[str, float]
    feature_order: list[str]


class PredictionResult(BaseModel):
    match_id: str
    team_a: str
    team_b: str
    team_a_probability: float
    team_b_probability: float
    timestamp: str
    model: str
    innings: int
    notes: Optional[str] = None
