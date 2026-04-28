import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "ipl_data"

def load_raw_data(data_dir=DATA_DIR):
    matches = pd.read_csv(f"{data_dir}/matches_updated_ipl_upto_2025.csv")
    deliveries = pd.read_csv(f"{data_dir}/deliveries_updated_ipl_upto_2025.csv")
    return matches, deliveries

def clean_matches(matches):
    matches = matches[matches["method"].isna()].copy()

    if "outcome" in matches.columns:
        matches = matches[~matches["outcome"].isin(["no result"])].copy()

    matches = matches.dropna(subset=["winner"])

    team_map = {
        "Delhi Daredevils": "Delhi Capitals",
        "Deccan Chargers": "Sunrisers Hyderabad",
        "Pune Warriors": "Rising Pune Supergiant",
        "Rising Pune Supergiants": "Rising Pune Supergiant",
        "Kings XI Punjab": "Punjab Kings",
    }

    for col in ["team1", "team2", "winner", "toss_winner"]:
        if col in matches.columns:
            matches[col] = matches[col].replace(team_map)

    keep = [c for c in [
        "matchId","season","city","venue",
        "team1","team2","toss_winner","toss_decision","winner"
    ] if c in matches.columns]

    return matches[keep]

def compute_targets(deliveries):
    first = deliveries[deliveries["inning"] == 1].copy()
    first["total_runs"] = first["batsman_runs"] + first["extras"].fillna(0)

    return (
        first.groupby("matchId")["total_runs"]
        .sum()
        .add(1)
        .rename("target")
    )

def merge_and_filter(matches, deliveries):
    df = deliveries.merge(matches, on="matchId", how="inner")
    return df[df["inning"] == 2].copy()

def build_match_state(df, target_map):
    df = df.sort_values(["matchId", "over", "ball"]).copy()

    df["total_runs"] = df["batsman_runs"] + df["extras"].fillna(0)
    df["runs_so_far"] = df.groupby("matchId")["total_runs"].cumsum()

    df["is_wicket"] = df["player_dismissed"].notna().astype(int)
    df["wickets_fallen"] = df.groupby("matchId")["is_wicket"].cumsum()

    df["is_legal"] = (df["isWide"].isna() & df["isNoBall"].isna()).astype(int)
    df["balls_bowled"] = df.groupby("matchId")["is_legal"].cumsum()

    df["overs_completed"] = df["balls_bowled"] / 6
    df["balls_remaining"] = (120 - df["balls_bowled"]).clip(lower=0)
    df["overs_remaining"] = df["balls_remaining"] / 6

    df["target"] = df["matchId"].map(target_map)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    df["required_runs"] = (df["target"] - df["runs_so_far"]).clip(lower=0)

    return df

def add_rate_features(df):
    df["current_run_rate"] = np.where(
        df["overs_completed"] > 0,
        df["runs_so_far"] / df["overs_completed"],
        0.0
    )

    df["required_run_rate"] = np.where(
        df["overs_remaining"] > 0,
        df["required_runs"] / df["overs_remaining"],
        df["required_runs"] * 6
    )

    df["run_rate_diff"] = df["current_run_rate"] - df["required_run_rate"]
    df["wickets_in_hand"] = 10 - df["wickets_fallen"]

    df["resources_remaining"] = (
        (df["balls_remaining"] / 120) *
        (df["wickets_in_hand"] / 10)
    )

    return df

def add_context_features(df):
    from sklearn.preprocessing import LabelEncoder

    df["toss_won_by_batting_team"] = (
        (df["toss_winner"] == df["batting_team"]) &
        (df["toss_decision"] == "field")
    ).astype(int)

    cat_cols = [c for c in [
        "batting_team","bowling_team","venue","city",
        "toss_winner","toss_decision"
    ] if c in df.columns]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders

def add_match_phase(df):
    df["match_phase"] = np.select(
        [df["overs_completed"] <= 6,
         df["overs_completed"] <= 15],
        ["powerplay", "middle"],
        default="death"
    )

    df["match_phase_enc"] = df["match_phase"].map({
        "powerplay": 0, "middle": 1, "death": 2
    })

    return df

def add_venue_features(df):
    match_level = (
        df.groupby("matchId")
        .agg(batting_team=("batting_team","first"),
             winner=("winner","first"),
             venue=("venue","first"))
        .reset_index()
    )

    match_level["chase_won"] = (
        match_level["batting_team"] == match_level["winner"]
    ).astype(int)

    venue_stats = (
        match_level.groupby("venue")["chase_won"]
        .agg(venue_matches="count", venue_chase_wins="sum")
        .reset_index()
    )

    venue_stats["venue_chase_win_rate"] = (
        venue_stats["venue_chase_wins"] /
        venue_stats["venue_matches"]
    ).round(3)

    venue_stats.loc[
        venue_stats["venue_matches"] < 5,
        "venue_chase_win_rate"
    ] = 0.5

    df = df.merge(
        venue_stats[["venue","venue_chase_win_rate","venue_matches"]],
        on="venue", how="left"
    )

    df["venue_chase_win_rate"] = df["venue_chase_win_rate"].fillna(0.5)

    return df

def add_target_variable(df):
    df["target_win"] = (df["batting_team"] == df["winner"]).astype(int)
    return df

def save_dataset(df, output_path="ipl_phase1.csv"):
    key_cols = [
        "runs_so_far","wickets_fallen","target",
        "required_runs","balls_remaining",
        "current_run_rate","required_run_rate","target_win"
    ]

    df = df.dropna(subset=key_cols)

    final_cols = [
        "matchId","season","over","ball",
        "runs_so_far","wickets_fallen","wickets_in_hand",
        "balls_bowled","balls_remaining",
        "overs_completed","overs_remaining",
        "target","required_runs",
        "current_run_rate","required_run_rate",
        "run_rate_diff","resources_remaining",
        "match_phase","match_phase_enc",
        "venue","venue_chase_win_rate","venue_matches",
        "batting_team","bowling_team",
        "batting_team_enc","bowling_team_enc",
        "toss_won_by_batting_team",
        "target_win"
    ]

    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    df.to_csv(output_path, index=False)
    return df

def run_phase1(data_dir=DATA_DIR, output_path="ipl_phase1.csv"):
    matches, deliveries = load_raw_data(data_dir)
    matches = clean_matches(matches)
    target_map = compute_targets(deliveries)

    df = merge_and_filter(matches, deliveries)
    df = build_match_state(df, target_map)
    df = add_rate_features(df)
    df, encoders = add_context_features(df)
    df = add_match_phase(df)
    df = add_venue_features(df)
    df = add_target_variable(df)
    df = save_dataset(df, output_path)

    return df, encoders

if __name__ == "__main__":
    df, encoders = run_phase1()