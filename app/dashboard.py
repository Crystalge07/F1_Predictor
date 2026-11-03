"""A very simple dashboard for looking at predictions vs. results.

Run it (from the repo root, with the virtual environment active):
    streamlit run app/dashboard.py

What it shows:
    1. Pick a season and a race in the sidebar.
    2. A table: every driver's starting grid slot, the model's predicted
       finishing position, and where they actually finished.
    3. The four metrics for that race, model vs. the grid-order baseline.
    4. A season summary: the same metrics for every race in the season.

How the predictions are made: walk-forward, exactly like the training script.
For the chosen race we train a model on every race BEFORE it, then predict it.
So what you see is what the model would really have said on that Saturday,
not a model that has already seen the answer.

This is a throwaway "look at the numbers" page, not the PRD's four-page
dashboard (that comes in Phase 6). Nothing is saved; everything is computed
when you open the page, and cached so clicking around stays fast.
"""

import pandas as pd
import streamlit as st

from f1predictor.db.connection import get_connection
from f1predictor.evaluate.metrics import score_race
from f1predictor.features.post_quali import build_post_quali_features
from f1predictor.models.baselines import actual_result, grid_order
from f1predictor.models.post_quali import predict_order, train_model

# The 2026 weight chosen by scripts/train_post_quali.py. If you re-tune it,
# update it here too.
WEIGHT_2026 = 3


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
# @st.cache_data remembers a function's return value for the same arguments.
# Streamlit re-runs this whole file on every click, so without the cache we'd
# rebuild the features (and retrain models) every time you pick a race.

@st.cache_data
def load_features() -> pd.DataFrame:
    """The full post-quali feature table (one row per driver per race)."""
    conn = get_connection()
    try:
        return build_post_quali_features(conn)
    finally:
        conn.close()


@st.cache_data
def load_names() -> tuple[dict[str, str], dict[str, str], pd.DataFrame]:
    """Lookups for display: driver_id -> name, constructor_id -> team name, and the race list."""
    conn = get_connection()
    try:
        drivers = {row["driver_id"]: f"{row['given_name']} {row['family_name']}"
                   for row in conn.execute("SELECT driver_id, given_name, family_name FROM drivers")}
        teams = {row["constructor_id"]: row["name"]
                 for row in conn.execute("SELECT constructor_id, name FROM constructors")}
        # Race sessions with their GP name, for the sidebar picker.
        races = pd.read_sql_query(
            """
            SELECT s.session_id, e.season, e.round, e.name
            FROM sessions s JOIN events e ON e.event_id = s.event_id
            WHERE s.session_type = 'R'
            """,
            conn,
        )
        return drivers, teams, races
    finally:
        conn.close()


@st.cache_data
def predict_race(session_id: int) -> dict:
    """Model order, grid order and actual result for one race (walk-forward)."""
    features = load_features()
    race = features[features["session_id"] == session_id]
    # Train ONLY on races strictly before this one (same rule as walk_forward.py).
    history = features[features["race_date"] < race["race_date"].iloc[0]]
    model = train_model(history, WEIGHT_2026)

    conn = get_connection()
    try:
        return {
            "model": predict_order(model, race),
            "grid": grid_order(conn, session_id),
            "actual": actual_result(conn, session_id),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="F1 Predictor", layout="wide")
st.title("F1 Predictor: predictions vs. results")
st.caption(f"Post-qualifying XGBoost model, walk-forward (2026 weight = {WEIGHT_2026}), "
           "compared with predicting the starting grid.")

features = load_features()
drivers, teams, all_races = load_names()

# Only races the feature table covers (i.e. races that have results).
races = (all_races[all_races["session_id"].isin(features["session_id"].unique())]
         .sort_values(["season", "round"]))

# --- Sidebar: choose a race --------------------------------------------------
seasons = sorted(races["season"].unique(), reverse=True)  # newest first
season = st.sidebar.selectbox("Season", seasons)
season_races = races[races["season"] == season]
# Show "R5 · Miami Grand Prix" but return the session_id behind it.
labels = {row.session_id: f"R{row.round} · {row.name}" for row in season_races.itertuples()}
session_id = st.sidebar.selectbox(
    "Race", list(labels), index=len(labels) - 1,  # default: latest race
    format_func=labels.get,
)

# --- One race ----------------------------------------------------------------
st.header(labels[session_id])
with st.spinner("Training on every earlier race..."):
    result = predict_race(session_id)

# Turn the two orders into {driver: predicted position}.
model_pos = {d: i + 1 for i, d in enumerate(result["model"])}
grid_pos = {d: i + 1 for i, d in enumerate(result["grid"])}
actual = result["actual"]

table = pd.DataFrame({
    "Driver": [drivers.get(d, d) for d in result["model"]],
    "Team": [teams.get(c, c) for c in
             features[features["session_id"] == session_id]
             .set_index("driver_id").loc[result["model"], "constructor_id"]],
    "Grid": [grid_pos[d] for d in result["model"]],
    "Predicted": [model_pos[d] for d in result["model"]],
    "Actual": [actual[d] for d in result["model"]],
})
# Positive = finished worse than predicted, negative = better.
table["Off by"] = table["Actual"] - table["Predicted"]

left, right = st.columns([3, 2])
with left:
    st.subheader("Predicted order (model)")
    st.dataframe(table, hide_index=True, width="stretch", height=740)

with right:
    st.subheader("How did it do?")
    model_score = score_race(result["model"], actual)
    grid_score = score_race(result["grid"], actual)
    metrics = pd.DataFrame(
        {
            "Model": [model_score.position_mae, model_score.spearman,
                      model_score.winner_hit, model_score.podium_hits],
            "Grid baseline": [grid_score.position_mae, grid_score.spearman,
                              grid_score.winner_hit, grid_score.podium_hits],
        },
        index=["Position MAE (lower = better)", "Spearman (higher = better)",
               "Winner right (1 = yes)", "Podium hits"],
    )
    st.dataframe(metrics.round(3), width="stretch")

# --- Season summary ----------------------------------------------------------
st.header(f"{season} season")
if st.checkbox("Score every race this season (trains one model per race, takes a few seconds)"):
    rows = []
    progress = st.progress(0.0)
    for i, row in enumerate(season_races.itertuples()):
        r = predict_race(row.session_id)
        m, g = score_race(r["model"], r["actual"]), score_race(r["grid"], r["actual"])
        rows.append({"Race": f"R{row.round} · {row.name}",
                     "Model MAE": m.position_mae, "Grid MAE": g.position_mae,
                     "Model winner": bool(m.winner_hit), "Grid winner": bool(g.winner_hit)})
        progress.progress((i + 1) / len(season_races))
    summary = pd.DataFrame(rows)
    st.dataframe(summary.round(2), hide_index=True, width="stretch")
    st.write(f"Average MAE: model **{summary['Model MAE'].mean():.2f}**, "
             f"grid **{summary['Grid MAE'].mean():.2f}**")
