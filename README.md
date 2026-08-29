# 🏏 IPL Real-Time Win Prediction

An end-to-end machine learning application that predicts the win probability of a chasing team during an IPL match.

The project combines an **LSTM-based sequence model**, **FastAPI**, and **Next.js** to provide an interactive match replay experience where users can move through deliveries and observe how the predicted win probability changes throughout the innings.

> **Note:** The current deployed version uses historical IPL data for replay and demonstration. It does not claim to provide live IPL predictions when no live ball-by-ball data source is available.

---

## 🚀 Features

- 🏏 IPL match replay using historical ball-by-ball data
- 🧠 LSTM-based win probability prediction
- 📈 Win and loss probability after each delivery
- 📊 Match state information including:
  - Score
  - Wickets
  - Balls bowled
  - Balls remaining
  - Required runs
  - Current run rate
  - Required run rate
- 🔄 Interactive delivery-by-delivery replay
- ⚡ FastAPI backend for model inference
- ⚛️ Next.js frontend
- ☁️ Designed for Vercel + Render deployment
- 🔐 Environment-variable based configuration

---

## 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │      Next.js UI      │
                    │      (Vercel)        │
                    └──────────┬───────────┘
                               │
                               │ HTTP / JSON
                               ▼
                    ┌──────────────────────┐
                    │      FastAPI         │
                    │      (Render)        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    LSTM Predictor    │
                    │                      │
                    │  lstm_model.pt       │
                    │  lstm_scaler.pkl     │
                    │  lstm_features.pkl   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Historical IPL Data  │
                    │   demo_matches.csv   │
                    └──────────────────────┘