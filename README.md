# IPL Real-Time Win Probability Predictor

A machine learning application that predicts the probability of a team winning an IPL chase as the match progresses.

The project combines a trained **XGBoost model**, historical IPL data, and live cricket match information to turn the current state of a match into a win probability.

> **Current status:** The application and API integration are implemented and unit-tested using mocked cricket API responses. The live CricketData API integration still needs to be verified with a real API key.

## What it does

The idea is simple:

**Give the model the current state of an IPL chase → predict the probability of each team winning.**

The application is designed to automatically obtain the current match state rather than requiring the user to manually enter runs, wickets, and overs.

The dashboard displays:

* Current score and wickets
* Overs completed
* Target and runs required
* Current Run Rate (CRR)
* Required Run Rate (RRR)
* Win probability for both teams
* Automatic updates as the match progresses

## How it works

```text
                CricketData.org
                      │
                      ▼
             Live Match Information
                      │
                      ▼
              Feature Engineering
                      │
                      ▼
             XGBoost Win Predictor
                      │
                      ▼
                Win Probability
                      │
                      ▼
               FastAPI Backend
                      │
                      ▼
              Web Dashboard
```

The application is designed to run on **Vercel**, with the frontend served as a static site and the FastAPI backend running as a Python serverless function.

## Machine Learning Model

The original project contains multiple trained models:

* Phase-1 XGBoost
* Logistic Regression
* Phase-3 XGBoost with additional momentum features
* LSTM for ball-by-ball sequences

For the deployed application, I use the **Phase-1 XGBoost model (`models/xgb.pkl`)**.

The main reason is that the production model needs to work with the information available from a single live scorecard snapshot.

The LSTM and Phase-3 models require richer ball-by-ball history to reliably reproduce their training features. Since the current API integration is based primarily on match/scorecard information, XGBoost provides a simpler and more reliable inference pipeline.

This is therefore a **deployment and data-availability decision**, rather than a claim that XGBoost is inherently better than the other models.

## Features Used

The XGBoost model expects 18 features:

```text
runs_so_far
wickets_fallen
wickets_in_hand
balls_bowled
balls_remaining
overs_completed
overs_remaining
target
required_runs
current_run_rate
required_run_rate
run_rate_diff
resources_remaining
match_phase_enc
venue_chase_win_rate
batting_team_enc
bowling_team_enc
toss_won_by_batting_team
```

The feature engineering layer converts the live match state into this exact feature order before passing it to the trained model.

## Live Cricket Data

The application uses **CricketData.org (formerly CricAPI)** as its cricket data provider.

The integration is isolated inside:

```text
services/cricket_api.py
```

This keeps the rest of the application independent of the provider's raw JSON format.

The API layer handles:

* API authentication
* Current matches
* Match information
* Scorecards
* Response normalization
* Timeouts
* Authentication errors
* Invalid match IDs
* Provider/API errors

The API key is stored as an environment variable and is never committed to the repository.

## Backend

The backend is built with **FastAPI**.

### API Endpoints

| Endpoint                      | Purpose                                  |
| ----------------------------- | ---------------------------------------- |
| `GET /api/health`             | Check whether the API is running         |
| `GET /api/matches`            | Get available current/recent matches     |
| `GET /api/matches/{match_id}` | Get normalized information about a match |
| `GET /api/predict/{match_id}` | Generate a win-probability prediction    |

Example prediction response:

```json
{
  "match_id": "abc123",
  "team_a": "Chennai Super Kings",
  "team_b": "Mumbai Indians",
  "team_a_probability": 0.7143,
  "team_b_probability": 0.2857,
  "innings": 2,
  "model": "xgboost-phase1"
}
```

The probabilities always sum to 1.

If a match has not reached the second innings, the prediction endpoint returns `409`, since the current model is designed for IPL chase predictions.

## Project Structure

```text
IPL-real-time-win-prediction/
│
├── api/
│   └── index.py
│
├── services/
│   ├── cricket_api.py
│   ├── feature_engineering.py
│   ├── predictor.py
│   └── schemas.py
│
├── artifacts/
│   ├── team_encoding.json
│   └── venue_win_rates.json
│
├── models/
│   ├── xgb.pkl
│   ├── xgb_phase3.pkl
│   ├── lr.pkl
│   ├── lstm_model.pt
│   └── ...
│
├── frontend/
│   └── index.html
│
├── tests/
│
├── ipl_data/
├── notebooks/
│
├── requirements.txt
├── vercel.json
├── .env.example
└── README.md
```

The large IPL datasets and notebooks are used for training and experimentation. They are not required by the deployed prediction service.

## Running Locally

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
CRICKET_API_KEY=your_api_key
```

Then start the FastAPI server:

```bash
uvicorn api.index:app --reload --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Deployment

The project is designed to be deployed on **Vercel**.

The deployment architecture is:

```text
Vercel
│
├── Frontend
│     └── Static dashboard
│
└── Python Serverless Function
      └── FastAPI
           ├── CricketData API
           ├── Feature Engineering
           └── XGBoost Model
```

### Deploying

1. Push the repository to GitHub.
2. Import the repository into Vercel.
3. Add the environment variable:

```text
CRICKET_API_KEY=your_api_key
```

4. Deploy the project.
5. Test:

```text
https://your-app.vercel.app/
https://your-app.vercel.app/api/health
```

The training datasets are excluded from the deployment bundle using `.vercelignore`.

## Testing

The project includes tests for:

* Feature engineering
* Model prediction
* Cricket API response parsing
* API error handling
* FastAPI endpoints

External cricket API requests are mocked during unit testing so tests do not depend on the availability of the external service.

## Limitations

There are a few things I am intentionally keeping transparent.

### Live API verification

The application has not yet been tested end-to-end with a real CricketData API key.

The API client is implemented against the documented response structure and tested with mocked responses, but the actual live response needs to be verified.

### Team encoding

The original training pipeline did not persist the `LabelEncoder` used for team names.

A compatible team-encoding artifact was therefore reconstructed from the existing dataset and saved in:

```text
artifacts/team_encoding.json
```

If the training dataset changes, this mapping should be regenerated.

### Historical team names

Some IPL franchises have changed names over the years. The original dataset contains inconsistencies between historical team names, so the project includes aliases for known renamed teams.

A future version should clean this at the dataset level and retrain the model.

### Venue matching

Venue names from the live API may not exactly match the historical dataset.

The feature engineering layer first attempts an exact match, then a looser venue match, and finally falls back to a neutral win rate when no match is found.

### Model compatibility

The XGBoost model was trained using an older XGBoost version. The current environment may produce a compatibility warning when loading the pickle.

If this becomes a hard compatibility issue, the model should be re-saved using XGBoost's native model format.

## Future Improvements

Some improvements I would like to add:

* Integrate a reliable ball-by-ball data source
* Add a rolling ball-history buffer
* Enable the Phase-3 model's momentum features
* Experiment with the LSTM for sequence-based predictions
* Clean historical IPL team names and retrain
* Persist preprocessing artifacts directly during training
* Add short-term API response caching
* Improve probability calibration
* Add historical prediction visualizations
* Track prediction accuracy throughout a live season

## Why I Built This

I wanted to move beyond a traditional ML project where a model is trained once and predictions are made from a static dataset.

The goal was to build the complete pipeline:

**historical data → machine learning model → live data → feature engineering → API → prediction → web application → cloud deployment**

This project therefore focuses not only on the model itself, but also on the engineering required to turn an ML model into a usable application.
