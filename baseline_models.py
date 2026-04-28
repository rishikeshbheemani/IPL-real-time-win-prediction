import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, brier_score_loss
import xgboost as xgb


FEATURE_COLS = [
    "runs_so_far","wickets_fallen","wickets_in_hand","balls_bowled",
    "balls_remaining","overs_completed","overs_remaining","target",
    "required_runs","current_run_rate","required_run_rate","run_rate_diff",
    "resources_remaining","match_phase_enc","venue_chase_win_rate",
    "batting_team_enc","bowling_team_enc","toss_won_by_batting_team",
]
TARGET = "target_win"
TEST_SEASONS = ["2024", "2025"]


def load_data(path="ipl_phase1.csv"):
    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)

    test = df[df["season"].isin(TEST_SEASONS)].copy()
    train = df[~df["season"].isin(TEST_SEASONS)].copy()

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET].values

    return X_train, y_train, X_test, y_test


def evaluate(y_true, y_pred, y_proba):
    print("Log Loss   :", round(log_loss(y_true, y_proba), 4))
    print("Accuracy   :", round(accuracy_score(y_true, y_pred), 4))
    print("AUC-ROC    :", round(roc_auc_score(y_true, y_proba), 4))
    print("Brier Score:", round(brier_score_loss(y_true, y_proba), 4))
    print()


def train_logistic(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    print("Logistic Regression Results")
    evaluate(y_test, pred, proba)

    return model, scaler


def train_xgboost(X_train, y_train, X_test, y_test):
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    print("XGBoost Results")
    evaluate(y_test, pred, proba)

    return model


def save_models(lr_model, scaler, xgb_model):
    os.makedirs("models", exist_ok=True)

    pickle.dump({"model": lr_model, "scaler": scaler}, open("models/lr.pkl", "wb"))
    pickle.dump(xgb_model, open("models/xgb.pkl", "wb"))
    pickle.dump(FEATURE_COLS, open("models/features.pkl", "wb"))


def predict(state):
    xgb_model = pickle.load(open("models/xgb.pkl", "rb"))
    lr_bundle = pickle.load(open("models/lr.pkl", "rb"))
    cols = pickle.load(open("models/features.pkl", "rb"))

    X = np.array([[state[c] for c in cols]])

    lr = lr_bundle["model"].predict_proba(
        lr_bundle["scaler"].transform(X)
    )[0, 1]

    xgb = xgb_model.predict_proba(X)[0, 1]

    return {"xgboost": float(xgb), "logistic": float(lr)}


def main():
    X_train, y_train, X_test, y_test = load_data()

    lr_model, scaler = train_logistic(
        X_train, y_train, X_test, y_test
    )

    xgb_model = train_xgboost(
        X_train, y_train, X_test, y_test
    )

    save_models(lr_model, scaler, xgb_model)


if __name__ == "__main__":
    main()