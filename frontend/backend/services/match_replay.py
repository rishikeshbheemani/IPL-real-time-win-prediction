from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.services.lstm_features import SEQ_FEATURES
from backend.services.lstm_predictor import LSTMPredictor


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BACKEND_DIR / "data" / "demo_matches.csv"

MIN_SEQUENCE_LENGTH = 30


class MatchReplayError(Exception):
    """Raised when a replay operation cannot be completed."""


class MatchReplay:
    def __init__(self):
        self.data = self._load_data()
        self.predictor = LSTMPredictor()

    def _load_data(self) -> pd.DataFrame:
        if not DATA_FILE.exists():
            raise MatchReplayError(
                f"Replay dataset not found: {DATA_FILE}"
            )

        df = pd.read_csv(
            DATA_FILE,
            low_memory=False,
        )

        required_columns = {
            "matchId",
            "balls_bowled",
            "runs_so_far",
            "wickets_fallen",
            "target_win",
            "batting_team",
            "bowling_team",
            *SEQ_FEATURES,
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise MatchReplayError(
                f"Replay dataset is missing columns: {sorted(missing)}"
            )

        return df.sort_values(
            ["matchId", "balls_bowled"]
        ).reset_index(drop=True)

    def list_matches(self) -> list[dict]:
        """Return the matches available for replay."""

        matches = []

        for match_id, match in self.data.groupby("matchId"):
            first_row = match.iloc[0]

            # The dataset stores the teams as batting_team and
            # bowling_team rather than team1 and team2.
            batting_team = first_row.get("batting_team")
            bowling_team = first_row.get("bowling_team")

            matches.append(
                {
                    "match_id": str(match_id),
                    "deliveries": len(match),
                    "season": str(first_row.get("season", "")),
                    "team1": (
                        str(batting_team)
                        if pd.notna(batting_team)
                        else None
                    ),
                    "team2": (
                        str(bowling_team)
                        if pd.notna(bowling_team)
                        else None
                    ),
                }
            )

        return matches

    def get_match(self, match_id: str) -> pd.DataFrame:
        """Return all deliveries for one match."""

        match = self.data[
            self.data["matchId"].astype(str) == str(match_id)
        ].copy()

        if match.empty:
            raise MatchReplayError(
                f"Match '{match_id}' was not found."
            )

        return match.sort_values(
            "balls_bowled"
        ).reset_index(drop=True)

    def get_delivery(
        self,
        match_id: str,
        delivery_number: int,
    ) -> dict:
        """
        Return the match state and LSTM prediction after
        the selected delivery.
        """

        match = self.get_match(match_id)

        if delivery_number < 1:
            raise MatchReplayError(
                "Delivery number must be at least 1."
            )

        if delivery_number > len(match):
            raise MatchReplayError(
                f"Match only contains {len(match)} deliveries."
            )

        if delivery_number < MIN_SEQUENCE_LENGTH:
            raise MatchReplayError(
                f"LSTM prediction requires at least "
                f"{MIN_SEQUENCE_LENGTH} deliveries."
            )

        current_match = match.iloc[:delivery_number]

        sequence = current_match[SEQ_FEATURES].copy()

        probability = self.predictor.predict(sequence)

        current = current_match.iloc[-1]

        return {
            "match_id": str(match_id),
            "delivery_number": delivery_number,
            "total_deliveries": len(match),

            "score": int(current["runs_so_far"]),
            "wickets": int(current["wickets_fallen"]),

            "win_probability": probability,
            "loss_probability": 1.0 - probability,

            "balls_bowled": int(current["balls_bowled"]),
            "balls_remaining": int(
                current["balls_remaining"]
            ),

            "required_runs": int(
                current["required_runs"]
            ),

            "current_run_rate": float(
                current["current_run_rate"]
            ),

            "required_run_rate": float(
                current["required_run_rate"]
            ),
        }

    def get_replay_series(self, match_id: str) -> list[dict]:
        """
        Generate win probabilities for every delivery from the
        minimum sequence length to the end of the match.
        """

        match = self.get_match(match_id)

        if len(match) < MIN_SEQUENCE_LENGTH:
            raise MatchReplayError(
                f"Match contains only {len(match)} deliveries. "
                f"At least {MIN_SEQUENCE_LENGTH} are required."
            )

        predictions = []

        for delivery_number in range(
            MIN_SEQUENCE_LENGTH,
            len(match) + 1,
        ):
            current_match = match.iloc[:delivery_number]

            sequence = current_match[SEQ_FEATURES].copy()

            probability = self.predictor.predict(sequence)

            current = current_match.iloc[-1]

            predictions.append(
                {
                    "delivery_number": delivery_number,
                    "score": int(current["runs_so_far"]),
                    "wickets": int(current["wickets_fallen"]),
                    "win_probability": probability,
                    "loss_probability": 1.0 - probability,
                    "balls_bowled": int(
                        current["balls_bowled"]
                    ),
                    "balls_remaining": int(
                        current["balls_remaining"]
                    ),
                    "required_runs": int(
                        current["required_runs"]
                    ),
                    "current_run_rate": float(
                        current["current_run_rate"]
                    ),
                    "required_run_rate": float(
                        current["required_run_rate"]
                    ),
                }
            )

        return predictions