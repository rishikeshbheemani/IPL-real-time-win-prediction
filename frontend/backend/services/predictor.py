"""
Inference layer.

Model choice: the phase-1 XGBoost classifier (models/xgb.pkl), NOT the LSTM
(models/lstm_model.pt) and NOT the phase-3 XGBoost (models/xgb_phase3.pkl).

Why (see README "ML Model" section for the full writeup):
  1. Feature compatibility with live data -- the LSTM was trained on full
     variable-length ball-by-ball sequences (minimum 30 balls) per innings,
     scaled with a StandardScaler fit over the whole sequence. Reproducing
     that at inference time would require reconstructing and re-scaling the
     entire innings' delivery history from the live API on every request.
     CricketData's entry-tier endpoints expose match/scorecard snapshots,
     not a guaranteed complete ball-by-ball feed, so this input cannot be
     reliably built live without a higher-tier data source. Phase-3 XGBoost
     has the same problem in miniature: several of its features
     (runs_last_6, runs_last_12, rr_acceleration, etc.) are rolling windows
     over the last 6-12 deliveries, which also requires a ball-by-ball feed.
  2. The phase-1 model's 18 features are all derivable from a single
     current-state snapshot (score, wickets, overs, target, teams, toss,
     venue) -- exactly what the scorecard endpoint reliably returns.
  3. Model size / cold start: xgb.pkl is ~650KB vs. a PyTorch runtime plus
     weights for the LSTM. Vercel's Python serverless functions have a
     50MB (compressed) deployment limit and cold-start budget; XGBoost is a
     much safer fit than bundling torch.
  4. Predictive performance: baseline_models.py's own evaluation shows the
     XGBoost model outperforming the plain logistic-regression baseline on
     the held-out 2024-2025 seasons. We did not have a like-for-like AUC
     comparison against the LSTM/phase-3 model at matched inference
     conditions (they need ball history the live path can't supply), so
     this decision is driven primarily by (1)-(3), not a head-to-head
     accuracy contest -- documented here rather than glossed over.

If ball-by-ball access is added later (a higher CricketData tier, or a
different provider), the phase-3 or LSTM model can be swapped in by adding
a ball-history buffer to feature_engineering.py -- the FastAPI routes and
schemas do not need to change.
"""
from __future__ import annotations

import pickle
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from services.feature_engineering import FEATURE_ORDER, FeatureEngineeringError, build_features
from services.schemas import MatchState, PredictionResult

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_model = None
_model_lock = threading.Lock()


class PredictionError(Exception):
    pass


def _get_model():
    """Lazy, cached model load. In a serverless function this runs once per
    warm container instance, not once per request."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # re-check inside the lock
                with open(MODELS_DIR / "xgb.pkl", "rb") as f:
                    _model = pickle.load(f)
    return _model


def predict(state: MatchState) -> PredictionResult:
    try:
        features, warnings = build_features(state)
    except FeatureEngineeringError as exc:
        raise PredictionError(str(exc)) from exc

    model = _get_model()
    x = np.array([[features[c] for c in FEATURE_ORDER]])

    batting_win_prob = float(model.predict_proba(x)[0, 1])
    batting_win_prob = min(max(batting_win_prob, 0.0), 1.0)
    bowling_win_prob = 1.0 - batting_win_prob

    notes = "; ".join(warnings) if warnings else None

    return PredictionResult(
        match_id=state.match_id,
        team_a=state.batting_team,
        team_b=state.bowling_team,
        team_a_probability=round(batting_win_prob, 4),
        team_b_probability=round(bowling_win_prob, 4),
        timestamp=datetime.now(timezone.utc).isoformat(),
        model="xgboost-phase1",
        innings=state.innings,
        notes=notes,
    )
