"""
Transforms a normalized MatchState (from services/cricket_api.py) into the
EXACT feature vector, in the EXACT order, expected by the trained phase-1
XGBoost model (models/xgb.pkl + models/features.pkl), which was produced by
baseline_models.py / datasetup.py.

Feature order (from models/features.pkl):
    runs_so_far, wickets_fallen, wickets_in_hand, balls_bowled,
    balls_remaining, overs_completed, overs_remaining, target,
    required_runs, current_run_rate, required_run_rate, run_rate_diff,
    resources_remaining, match_phase_enc, venue_chase_win_rate,
    batting_team_enc, bowling_team_enc, toss_won_by_batting_team

Two features require artifacts that were NOT saved during training and had
to be reconstructed here (see README "Known limitations" for detail):

- batting_team_enc / bowling_team_enc: datasetup.py fit a fresh
  sklearn LabelEncoder every run and never persisted it. We rebuilt the
  exact mapping by re-running the same encoding step against the same
  training data and saved it to artifacts/team_encoding.json. This is
  correct as long as models/xgb.pkl was trained from the CSVs currently in
  ipl_data/ -- if that dataset changes, this mapping must be regenerated
  (see scripts note in README).
- venue_chase_win_rate: originally computed inline from the historical
  matches dataset. We precomputed it once into
  artifacts/venue_win_rates.json so production inference never needs to
  load the 89MB ipl_data CSVs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from services.schemas import MatchState

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

FEATURE_ORDER = [
    "runs_so_far", "wickets_fallen", "wickets_in_hand", "balls_bowled",
    "balls_remaining", "overs_completed", "overs_remaining", "target",
    "required_runs", "current_run_rate", "required_run_rate", "run_rate_diff",
    "resources_remaining", "match_phase_enc", "venue_chase_win_rate",
    "batting_team_enc", "bowling_team_enc", "toss_won_by_batting_team",
]

DEFAULT_VENUE_WIN_RATE = 0.5
UNKNOWN_TEAM_ENC = -1  # explicit sentinel; see build_features() note below


def _load_json(name: str) -> dict:
    path = ARTIFACTS_DIR / name
    with open(path, "r") as f:
        return json.load(f)


_TEAM_ENCODING = _load_json("team_encoding.json")
_VENUE_WIN_RATES = _load_json("venue_win_rates.json")

# Franchises that have been renamed since the historical training data was
# recorded. The live API will report the current name; we map it back to
# whichever label the training data used, so we hit a real encoded class
# instead of an unseen one. See README limitations for why this exists.
_TEAM_NAME_ALIASES = {
    "Delhi Capitals": "Delhi Capitals",
    "Punjab Kings": "Punjab Kings",
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
    "Royal Challengers Bangalore": "Royal Challengers Bangalore",
    "Sunrisers Hyderabad": "Sunrisers Hyderabad",
}


class FeatureEngineeringError(Exception):
    """Raised when a MatchState cannot be turned into a valid feature
    vector (e.g. first innings, no target yet)."""


def _encode_team(name: str) -> int:
    canonical = _TEAM_NAME_ALIASES.get(name, name)
    if canonical in _TEAM_ENCODING:
        return _TEAM_ENCODING[canonical]
    # Unseen team name (e.g. a franchise not present in training data).
    # We do NOT invent a plausible-looking encoding; we fall back to the
    # median known class so the vector is still valid, and the caller is
    # told via PredictionResult.notes that this happened.
    return int(round(sorted(_TEAM_ENCODING.values())[len(_TEAM_ENCODING) // 2]))


def _venue_win_rate(venue: str | None) -> float:
    if not venue:
        return DEFAULT_VENUE_WIN_RATE
    if venue in _VENUE_WIN_RATES:
        return _VENUE_WIN_RATES[venue]
    # Try a loose match (live API venue strings often include extra city
    # suffixes, e.g. "Wankhede Stadium, Mumbai" vs "Wankhede Stadium").
    for known_venue, rate in _VENUE_WIN_RATES.items():
        if known_venue.split(",")[0].strip() == venue.split(",")[0].strip():
            return rate
    return DEFAULT_VENUE_WIN_RATE


def build_features(state: MatchState) -> tuple[dict[str, float], list[str]]:
    """Returns (features_by_name, warnings)."""
    if state.target is None:
        raise FeatureEngineeringError(
            "Match is still in the first innings; no target/chase to predict "
            "a win probability for yet."
        )

    warnings: list[str] = []

    wickets_in_hand = 10 - state.wickets_fallen
    balls_remaining = max(0, 120 - state.balls_bowled)
    overs_completed = state.balls_bowled / 6
    overs_remaining = balls_remaining / 6

    required_runs = max(0, state.target - state.runs_so_far)

    current_run_rate = (state.runs_so_far / overs_completed) if overs_completed > 0 else 0.0
    if overs_remaining > 0:
        required_run_rate = required_runs / overs_remaining
    else:
        required_run_rate = required_runs * 6

    run_rate_diff = current_run_rate - required_run_rate
    resources_remaining = (balls_remaining / 120) * (wickets_in_hand / 10)

    if overs_completed <= 6:
        match_phase_enc = 0  # powerplay
    elif overs_completed <= 15:
        match_phase_enc = 1  # middle
    else:
        match_phase_enc = 2  # death

    venue_rate = _venue_win_rate(state.venue)
    if state.venue and venue_rate == DEFAULT_VENUE_WIN_RATE and state.venue not in _VENUE_WIN_RATES:
        warnings.append(f"Venue '{state.venue}' not in historical data; using neutral 0.5 win rate.")

    if state.batting_team not in _TEAM_ENCODING and _TEAM_NAME_ALIASES.get(state.batting_team) not in _TEAM_ENCODING:
        warnings.append(f"Batting team '{state.batting_team}' unseen in training data.")
    if state.bowling_team not in _TEAM_ENCODING and _TEAM_NAME_ALIASES.get(state.bowling_team) not in _TEAM_ENCODING:
        warnings.append(f"Bowling team '{state.bowling_team}' unseen in training data.")

    toss_won_by_batting_team = int(
        bool(state.toss_winner)
        and state.toss_winner == state.batting_team
        and (state.toss_decision or "").lower() == "field"
    )

    features = {
        "runs_so_far": float(state.runs_so_far),
        "wickets_fallen": float(state.wickets_fallen),
        "wickets_in_hand": float(wickets_in_hand),
        "balls_bowled": float(state.balls_bowled),
        "balls_remaining": float(balls_remaining),
        "overs_completed": float(overs_completed),
        "overs_remaining": float(overs_remaining),
        "target": float(state.target),
        "required_runs": float(required_runs),
        "current_run_rate": float(current_run_rate),
        "required_run_rate": float(required_run_rate),
        "run_rate_diff": float(run_rate_diff),
        "resources_remaining": float(resources_remaining),
        "match_phase_enc": float(match_phase_enc),
        "venue_chase_win_rate": float(venue_rate),
        "batting_team_enc": float(_encode_team(state.batting_team)),
        "bowling_team_enc": float(_encode_team(state.bowling_team)),
        "toss_won_by_batting_team": float(toss_won_by_batting_team),
    }

    assert list(features.keys()) == FEATURE_ORDER, "Feature order drifted from FEATURE_ORDER"

    return features, warnings
