"""The guardrail: alert when the live model falls behind its baseline.

Every live prediction is scored after its race next to that stage's
baseline (evaluate/scorer.py). This check looks at the most recent scored
live races and raises an alert if the model has been clearly worse than
the baseline over them. It only ALERTS (decision 2026-11-03): nothing is
swapped, retrained or switched off. A person reads the alert and decides.

The rule, per stage:

    take the last WINDOW scored live races of that stage;
    for each metric in METRICS, average the model and the baseline;
    alert if the model is worse than the baseline by more than MARGIN
    (as a share of the baseline).

The numbers (chosen 2026-11-03, on the 2026 backtest, docs/PRD.md):

    WINDOW = 8   With 3 or 5 races, ordinary bad luck already pushed some
                 stage's model 13-29% behind its baseline over a stretch. With
                 8, the worst stretch in 2026 was +10%. 8 races is about a
                 third of a season: longer would notice a real problem too late.
    MARGIN = 15% About 5 points above the worst normal stretch (+10%), as
                 headroom: those numbers are in-sample, live noise will be a
                 little larger. A real break still trips it fast: a model
                 suddenly 40% worse crosses 15% on the 8-race average after
                 3 races; a badly broken one (e.g. a shuffled order, ~2x the
                 baseline's places off) after 2.
    METRICS      places off (position_mae) and podium Brier (brier_podium):
                 the two steady ones, and the two things the model claims
                 to do (rank the field, price the podium). Not winner_hit /
                 podium_hits (one race moves them 33-100%), not brier_win (a
                 tiny baseline, ~0.03, so small differences look huge; and
                 post-quali is already behind on it in-sample).
                 Both are "lower is better".

Which runs count: live runs only (never backtest replays), all seasons
together, 2026's rehearsal included: this is a health check, not the
scorecard (the scorecard keeps them apart, evaluate/scorer.summary). So it
can first fire around round 4 of 2027. Re-check MARGIN after ~12 live 2027
races.

One alert per stage, metric and newest race in the window: the alert key
names that race, so it's raised once when that race is scored, not every
hour, and again after the next race if the model is still behind.
"""

import sqlite3

WINDOW = 8
MARGIN = 0.15
METRICS = ("position_mae", "brier_podium")     # both: lower is better
STAGES = ("pre_weekend", "practice", "post_quali")


def drift_alerts(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Every (alert_key, message) the rule raises right now. Pure read."""
    alerts = []
    for stage in STAGES:
        for metric in METRICS:
            # The last WINDOW scored live races of this stage, newest first.
            # Ordered by race date, then round (a race is only ever scored
            # once per stage: one live run per race and stage, db/schema.sql).
            rows = conn.execute(
                """
                SELECT e.season, e.round, s.model_value, s.baseline_value
                FROM scores s
                JOIN prediction_runs r ON r.run_id = s.run_id
                JOIN events e          ON e.event_id = r.event_id
                WHERE r.mode = 'live' AND r.stage = ? AND s.metric = ?
                  AND s.baseline_value IS NOT NULL
                ORDER BY e.race_date DESC, e.season DESC, e.round DESC
                LIMIT ?
                """, (stage, metric, WINDOW)).fetchall()
            if len(rows) < WINDOW:
                continue                     # too few races to tell luck from a problem
            model = sum(r["model_value"] for r in rows) / WINDOW
            baseline = sum(r["baseline_value"] for r in rows) / WINDOW
            if baseline <= 0:
                continue                     # a perfect baseline: no share to compare with
            # Lower is better, so model above baseline = worse. As a share of
            # the baseline: 0.20 means "20% worse than the baseline".
            worse = (model - baseline) / baseline
            if worse > MARGIN:
                newest, oldest = rows[0], rows[-1]
                alerts.append((
                    f"drift:{stage}:{metric}:{newest['season']}:{newest['round']}",
                    f"drift: {stage} {metric} is {worse:.0%} worse than its baseline over the last "
                    f"{WINDOW} live races ({oldest['season']} R{oldest['round']} to "
                    f"{newest['season']} R{newest['round']}: model {model:.3f} vs baseline {baseline:.3f}; "
                    f"alert threshold {MARGIN:.0%}). Nothing was changed automatically.",
                ))
    return alerts
