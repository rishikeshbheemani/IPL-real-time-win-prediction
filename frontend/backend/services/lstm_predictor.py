from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from backend.services.lstm_features import SEQ_FEATURES
from backend.services.lstm_model import LSTMModel


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


class LSTMPredictor:
    def __init__(self):
        self.device = torch.device("cpu")

        with open(MODEL_DIR / "lstm_scaler.pkl", "rb") as file:
            self.scaler = pickle.load(file)

        self.model = LSTMModel(input_size=len(SEQ_FEATURES))

        state_dict = torch.load(
            MODEL_DIR / "lstm_model.pt",
            map_location=self.device,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sequence: np.ndarray) -> float:
        if sequence.ndim != 2:
            raise ValueError(
                "Expected sequence with shape (timesteps, features)."
            )

        if sequence.shape[1] != len(SEQ_FEATURES):
            raise ValueError(
                f"Expected {len(SEQ_FEATURES)} features, "
                f"got {sequence.shape[1]}."
            )

        scaled = self.scaler.transform(sequence)

        x = torch.tensor(
            scaled,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        lengths = torch.tensor(
            [len(sequence)],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            logit = self.model(x, lengths)
            probability = torch.sigmoid(logit).item()

        return float(probability)


predictor = LSTMPredictor()