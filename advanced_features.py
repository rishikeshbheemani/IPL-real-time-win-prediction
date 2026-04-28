import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, roc_auc_score
import xgboost as xgb


BASE_FEATURES = [
    "runs_so_far","wickets_fallen","wickets_in_hand",
    "balls_bowled","balls_remaining","overs_completed","overs_remaining",
    "target","required_runs","current_run_rate","required_run_rate",
    "run_rate_diff","resources_remaining","match_phase_enc",
    "venue_chase_win_rate","batting_team_enc","bowling_team_enc",
    "toss_won_by_batting_team"
]

NEW_FEATURES = [
    "runs_last_6","runs_last_12","wickets_last_12",
    "boundaries_last_6","rr_acceleration",
    "pressure_index_a","pressure_index_b",
    "phase_x_rrr","phase_x_wkts"
]

ALL_FEATURES = BASE_FEATURES + NEW_FEATURES
TARGET = "target_win"
TEST_SEASONS = ["2024","2025"]


def load_data(path="ipl_phase1.csv"):
    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    return df.sort_values(["matchId","balls_bowled"]).reset_index(drop=True)


def add_ball_features(df):
    df["ball_runs"] = df.groupby("matchId")["runs_so_far"].diff().fillna(df["runs_so_far"]).clip(0)
    df["ball_wicket"] = df.groupby("matchId")["wickets_fallen"].diff().fillna(df["wickets_fallen"]).clip(0)
    df["is_boundary"] = (df["ball_runs"] >= 4).astype(int)
    return df


def add_momentum(df):
    df["runs_last_6"] = df.groupby("matchId")["ball_runs"].transform(lambda x: x.rolling(6,1).sum())
    df["runs_last_12"] = df.groupby("matchId")["ball_runs"].transform(lambda x: x.rolling(12,1).sum())
    df["wickets_last_12"] = df.groupby("matchId")["ball_wicket"].transform(lambda x: x.rolling(12,1).sum())
    df["boundaries_last_6"] = df.groupby("matchId")["is_boundary"].transform(lambda x: x.rolling(6,1).sum())
    df["rr_acceleration"] = df["runs_last_6"] - df["current_run_rate"]
    return df


def add_pressure(df):
    df["pressure_index_a"] = (df["required_run_rate"] / df["current_run_rate"].replace(0,np.nan)).clip(0,10).fillna(5)
    df["pressure_index_b"] = (
        (df["required_runs"] / df["balls_remaining"].replace(0,np.nan))
        * (1 + df["wickets_fallen"]/10)
    ).clip(0,15).fillna(10)
    return df


def add_interactions(df):
    df["phase_x_rrr"] = df["match_phase_enc"] * df["required_run_rate"]
    df["phase_x_wkts"] = df["match_phase_enc"] * df["wickets_fallen"]
    return df

def plot_feature_importance(model, feature_names, top_n=15):
    importance = model.get_booster().get_score(importance_type="gain")

    imp_df = pd.DataFrame({
        "feature": [f"f{i}" for i in range(len(feature_names))],
        "gain": [importance.get(f"f{i}", 0) for i in range(len(feature_names))]
    })

    imp_df["feature"] = feature_names
    imp_df = imp_df.sort_values("gain", ascending=False).head(top_n)

    plt.figure(figsize=(8,6))
    plt.barh(imp_df["feature"], imp_df["gain"])
    plt.xlabel("Gain")
    plt.title("Feature Importance (XGBoost)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show() 

def train_model(df):
    train = df[~df["season"].isin(TEST_SEASONS)]
    test  = df[df["season"].isin(TEST_SEASONS)]

    X_train = train[ALL_FEATURES].values
    y_train = train[TARGET].values
    X_test  = test[ALL_FEATURES].values
    y_test  = test[TARGET].values

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

    proba = model.predict_proba(X_test)[:,1]
    pred  = (proba >= 0.5).astype(int)

    print("Log Loss   :", round(log_loss(y_test, proba),4))
    print("Accuracy   :", round(accuracy_score(y_test, pred),4))
    print("AUC-ROC    :", round(roc_auc_score(y_test, proba),4))
    print("Brier Score:", round(brier_score_loss(y_test, proba),4))

    return model


def save(df, model):
    os.makedirs("models", exist_ok=True)
    df.to_csv("ipl_phase3.csv", index=False)

    pickle.dump(model, open("models/xgb_phase3.pkl","wb"))
    pickle.dump(ALL_FEATURES, open("models/features_phase3.pkl","wb"))


def predict(state):
    model = pickle.load(open("models/xgb_phase3.pkl","rb"))
    cols  = pickle.load(open("models/features_phase3.pkl","rb"))

    X = np.array([[state[c] for c in cols]])
    return float(model.predict_proba(X)[0,1])


def main():
    df = load_data()
    df = add_ball_features(df)
    df = add_momentum(df)
    df = add_pressure(df)
    df = add_interactions(df)

    model = train_model(df)
    plot_feature_importance(model, ALL_FEATURES)
    save(df, model)


if __name__ == "__main__":
    main()