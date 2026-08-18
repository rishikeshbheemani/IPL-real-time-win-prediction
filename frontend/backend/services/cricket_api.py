"""
Client for the CricketData.org (formerly CricAPI) live cricket data provider.
https://cricketdata.org/

Design notes
------------
- The API key is read ONLY from the environment variable CRICKET_API_KEY.
  It is never hardcoded and never returned in any response body.
- This module is the ONLY place that knows about CricketData's raw JSON
  shape. Everything downstream (feature_engineering.py, predictor.py, the
  FastAPI routes) works exclusively with the normalized models in
  services/schemas.py.
- The free/entry CricketData tier exposes match summaries and periodic
  scorecards (score per innings, overs, wickets) but NOT a granular
  ball-by-ball feed for every match. Because of that, this client extracts
  the *current match state* (score / overs / wickets / target) rather than
  a full delivery-by-delivery sequence. This is exactly what the phase-1
  XGBoost model needs (see predictor.py for why that model was chosen).
- All network calls have explicit timeouts and raise a small set of typed
  exceptions so the API layer can translate them into clean HTTP errors
  instead of leaking provider details.
"""
from __future__ import annotations

import os
import logging
from typing import Any, Optional

import requests

from services.schemas import MatchInfo, MatchState

logger = logging.getLogger("cricket_api")

BASE_URL = "https://api.cricketdata.org/v1"
# CricketData's API has historically also been reachable at api.cricapi.com/v1
# with an identical contract (CricAPI -> CricketData rebrand). If the primary
# host is unreachable, we fall back to it.
FALLBACK_BASE_URL = "https://api.cricapi.com/v1"

REQUEST_TIMEOUT_SECONDS = 8


class CricketAPIError(Exception):
    """Base class for all cricket-data-provider errors."""


class CricketAPIAuthError(CricketAPIError):
    """Raised when the API key is missing or rejected."""


class CricketAPITimeoutError(CricketAPIError):
    """Raised when the provider does not respond in time."""


class CricketAPIResponseError(CricketAPIError):
    """Raised when the provider responds but the payload is malformed
    or reports an application-level failure."""


class MatchNotFoundError(CricketAPIError):
    """Raised when a requested match_id does not exist / isn't returned."""


def _get_api_key() -> str:
    key = os.environ.get("CRICKET_API_KEY")
    if not key:
        raise CricketAPIAuthError(
            "CRICKET_API_KEY is not set. Add it to your .env locally, or to "
            "the Vercel project's Environment Variables in production."
        )
    return key


def _request(path: str, params: Optional[dict] = None) -> dict:
    params = dict(params or {})
    params["apikey"] = _get_api_key()

    last_error: Optional[Exception] = None
    for base in (BASE_URL, FALLBACK_BASE_URL):
        url = f"{base}{path}"
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.Timeout as exc:
            last_error = CricketAPITimeoutError(f"Timed out calling {path}")
            continue
        except requests.RequestException as exc:
            last_error = CricketAPIResponseError(f"Network error calling {path}: {exc}")
            continue

        if resp.status_code == 401 or resp.status_code == 403:
            raise CricketAPIAuthError("CricketData rejected the API key (401/403).")
        if resp.status_code >= 500:
            last_error = CricketAPIResponseError(
                f"CricketData returned {resp.status_code} for {path}"
            )
            continue
        if resp.status_code != 200:
            raise CricketAPIResponseError(
                f"CricketData returned unexpected status {resp.status_code} for {path}"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise CricketAPIResponseError(f"Non-JSON response from {path}") from exc

        # CricketData wraps every response with a top-level "status" field.
        if payload.get("status") not in (None, "success"):
            raise CricketAPIResponseError(
                f"CricketData reported failure for {path}: {payload.get('status')} "
                f"- {payload.get('info') or payload.get('message')}"
            )

        return payload

    raise last_error or CricketAPIResponseError(f"Failed to call {path}")


def _overs_to_balls(overs: float) -> int:
    """Convert cricket over notation (e.g. 12.3 = 12 overs, 3 balls) to a
    legal-ball count. Cricket overs are base-6 in the fractional part, so
    this cannot be done with plain float math."""
    if overs is None:
        return 0
    whole = int(overs)
    frac_balls = round((overs - whole) * 10)
    frac_balls = max(0, min(frac_balls, 5))
    return whole * 6 + frac_balls


def fetch_current_matches() -> list[MatchInfo]:
    """GET /currentMatches -- list of matches currently live or recently
    completed. Used to power GET /api/matches."""
    payload = _request("/currentMatches", {"offset": 0})
    raw_matches = payload.get("data") or []

    matches: list[MatchInfo] = []
    for m in raw_matches:
        teams = m.get("teams") or []
        matches.append(
            MatchInfo(
                match_id=m.get("id", ""),
                name=m.get("name", "Unknown match"),
                status=m.get("status", "unknown"),
                venue=m.get("venue"),
                date=m.get("date") or m.get("dateTimeGMT"),
                team1=teams[0] if len(teams) > 0 else None,
                team2=teams[1] if len(teams) > 1 else None,
                match_type=m.get("matchType"),
                is_live=bool(m.get("matchStarted")) and not bool(m.get("matchEnded")),
            )
        )
    return matches


def fetch_match_info(match_id: str) -> MatchInfo:
    payload = _request("/match_info", {"id": match_id})
    m = payload.get("data")
    if not m:
        raise MatchNotFoundError(f"No match found for id={match_id}")

    teams = m.get("teams") or []
    return MatchInfo(
        match_id=m.get("id", match_id),
        name=m.get("name", "Unknown match"),
        status=m.get("status", "unknown"),
        venue=m.get("venue"),
        date=m.get("date") or m.get("dateTimeGMT"),
        team1=teams[0] if len(teams) > 0 else None,
        team2=teams[1] if len(teams) > 1 else None,
        match_type=m.get("matchType"),
        is_live=bool(m.get("matchStarted")) and not bool(m.get("matchEnded")),
    )


def fetch_match_state(match_id: str) -> MatchState:
    """GET /match_scorecard -- current score/overs/wickets state, normalized
    into the second-innings match state our model consumes.

    Raises MatchNotFoundError, or CricketAPIResponseError if the match
    exists but hasn't reached an innings we can predict for (see
    predictor.py, which treats target=None as "no prediction yet").
    """
    payload = _request("/match_scorecard", {"id": match_id})
    m = payload.get("data")
    if not m:
        raise MatchNotFoundError(f"No match found for id={match_id}")

    score_entries: list[dict[str, Any]] = m.get("score") or []
    if not score_entries:
        raise CricketAPIResponseError(
            f"Match {match_id} has no score data yet (toss/pre-match)."
        )

    # score_entries look like: [{"r": 178, "w": 6, "o": 20.0, "inning": "Team A Inning 1"}, ...]
    current = score_entries[-1]
    innings_number = len(score_entries)  # 1 = first innings in progress, 2 = second

    target = None
    if innings_number >= 2:
        target = int(score_entries[0].get("r", 0)) + 1

    teams = m.get("teams") or []
    batting_team = current.get("inning", "").split(" Inning")[0].strip() or None
    if not batting_team and teams:
        batting_team = teams[innings_number - 1] if innings_number - 1 < len(teams) else teams[0]
    bowling_team = None
    for t in teams:
        if t != batting_team:
            bowling_team = t
            break

    balls_bowled = _overs_to_balls(float(current.get("o", 0) or 0))

    return MatchState(
        match_id=match_id,
        batting_team=batting_team or "Unknown",
        bowling_team=bowling_team or "Unknown",
        venue=m.get("venue"),
        toss_winner=m.get("tossWinner"),
        toss_decision=m.get("tossChoice") or m.get("tossDecision"),
        runs_so_far=int(current.get("r", 0)),
        wickets_fallen=min(int(current.get("w", 0)), 10),
        balls_bowled=min(balls_bowled, 120),
        target=target,
        innings=2 if innings_number >= 2 else 1,
        status=m.get("status", "live"),
    )
