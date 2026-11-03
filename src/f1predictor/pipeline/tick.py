"""The tick: one run of the pipeline. Plan, then do everything that's due.

The scheduler runs this every hour (Phase 5, step 5). Each tick:

    1. FETCH    plan, then fetch every session / Jolpica part that's ready
                (pipeline/fetch.py).
    2. PREDICT  plan again (the new data may have made a stage ready), then,
                for each stage due: make the prediction, run the live checks
                (predict/stage.live_problems: input sanity, cutoff not passed,
                no live run yet, clean code) and write EXACTLY the prediction
                that was checked. A prediction that fails a check is not
                logged: it's reported as blocked, and the next tick tries
                again while the cutoff hasn't passed.
                This is the automated version of "dry run first": the
                prediction is built and checked in memory, and only then written.
    3. SCORE    plan again, then score every race whose result is in.
    4. REPORT   record the tick in pipeline_ticks, and raise alerts for
                problems that matter: blocked predictions, stages missed,
                sessions given up on, Jolpica data late. (A single failed
                fetch isn't one: data a few minutes late is normal.) Each alert is raised once
                (pipeline_alerts); the tick returns how many are new, and the
                command-line script exits with an error if there are any,
                which is what makes GitHub Actions send an email.

Retraining: there is no separate step. Every prediction trains its stage's
model on every race before it (predict/stage.py), so once a race's result is
fetched, the next prediction has learned from it. That's the PRD's "append
the race to the training set, retrain", with nothing to forget to run.

dry_run=True: fetches data (safe: upserts, validated, idempotent) and builds
and checks predictions, but writes nothing to the prediction log or scores.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from f1predictor.evaluate.scorer import score_runs
from f1predictor.pipeline.fetch import run_fetches
from f1predictor.pipeline.plan import Action, plan, to_utc_text
from f1predictor.predict.stage import live_problems, make_prediction, write_run

logger = logging.getLogger(__name__)


@dataclass
class TickReport:
    """What one tick did. Every list holds short readable lines."""
    now: str
    dry_run: bool
    fetched: dict[str, str] = field(default_factory=dict)   # {item_key: status}
    predicted: list[str] = field(default_factory=list)      # logged (or, dry run, would be)
    blocked: list[str] = field(default_factory=list)        # predictions that failed a check
    scored: list[str] = field(default_factory=list)
    alerts: list[tuple[str, str]] = field(default_factory=list)   # (alert_key, message), all of them
    new_alerts: list[str] = field(default_factory=list)     # messages raised for the first time

    def lines(self) -> list[str]:
        out = [f"tick at {self.now}{' (DRY RUN: nothing logged)' if self.dry_run else ''}"]
        out += [f"  fetched  {key}: {status}" for key, status in self.fetched.items()]
        out += [f"  predict  {line}" for line in self.predicted]
        out += [f"  BLOCKED  {line}" for line in self.blocked]
        out += [f"  scored   {line}" for line in self.scored]
        out += [f"  ALERT    {message}" for message in self.new_alerts]
        if len(out) == 1:
            out.append("  nothing due")
        return out


def _label(action: Action) -> str:
    return f"{action.season} R{action.round} {action.target}"


def run_tick(conn: sqlite3.Connection, now: datetime | None = None, dry_run: bool = False,
             jolpica_client=None, openf1_client=None, n_sims: int | None = None) -> TickReport:
    """Run one tick at `now` (default: the real clock). See the module docstring.

    Clients default to real ones (pipeline/fetch.make_clients); tests pass fakes.
    n_sims overrides the number of simulated races (tests use fewer, for speed).
    """
    from f1predictor.pipeline.fetch import make_clients

    now = now or datetime.now(timezone.utc)
    if jolpica_client is None or openf1_client is None:
        jolpica_client, openf1_client = make_clients()
    report = TickReport(now=to_utc_text(now), dry_run=dry_run)
    tick_id = conn.execute("INSERT INTO pipeline_ticks (started_at_utc, dry_run) VALUES (?, ?)",
                           (to_utc_text(datetime.now(timezone.utc)), int(dry_run))).lastrowid
    conn.commit()

    # --- 1. Fetch ------------------------------------------------------------
    # A failed fetch is NOT an alert by itself: data that's a few minutes
    # late is normal, and the next tick retries. It shows in the report;
    # problems that last are alerted by the planner ('gave_up', 'late'), and
    # what really matters (a prediction blocked or missed) is alerted below.
    report.fetched = run_fetches(conn, plan(conn, now), now, jolpica_client, openf1_client)

    # --- 2. Predict ----------------------------------------------------------
    for action in plan(conn, now):
        if action.kind != "predict":
            continue
        try:
            prediction = make_prediction(conn, action.event_id, action.target,
                                         **({"n_sims": n_sims} if n_sims else {}))
            problems = live_problems(conn, prediction, now)
        except Exception as error:            # a prediction must never crash the tick
            prediction, problems = None, [f"prediction failed: {error}"]
        if problems:
            line = f"{_label(action)}: " + "; ".join(problems)
            report.blocked.append(line)
            report.alerts.append((f"blocked:{action.season}:{action.round}:{action.target}", f"not logged, {line}"))
            continue
        if dry_run:
            report.predicted.append(f"{_label(action)} (passed checks; not logged: dry run)")
        else:
            run_id = write_run(conn, prediction, "live")
            favourite = prediction.drivers.sort_values("p_win", ascending=False).iloc[0]
            report.predicted.append(f"{_label(action)} -> live run {run_id} "
                                    f"(favourite {favourite.driver_id} {favourite.p_win:.0%})")

    # --- 3. Score ------------------------------------------------------------
    for action in plan(conn, now):
        if action.kind != "score" or dry_run:
            continue
        run_ids = [r[0] for r in conn.execute(
            "SELECT run_id FROM prediction_runs r WHERE event_id = ? "
            "AND NOT EXISTS (SELECT 1 FROM scores s WHERE s.run_id = r.run_id)", (action.event_id,))]
        scored = score_runs(conn, run_ids)
        report.scored.append(f"{action.season} R{action.round}: {len(scored)} run(s)")

    # --- 4. Informational problems, then the alerts ------------------------------
    for action in plan(conn, now):
        if action.kind in ("missed", "gave_up", "late"):
            report.alerts.append((f"{action.kind}:{action.season}:{action.round}:{action.target}",
                                  f"{action.kind}: {_label(action)}: {action.reason}"))
    for key, message in report.alerts:
        # Dry runs only report; they don't mark an alert as already sent.
        seen = conn.execute("SELECT 1 FROM pipeline_alerts WHERE alert_key = ?", (key,)).fetchone()
        if seen:
            continue
        report.new_alerts.append(message)
        if not dry_run:
            conn.execute("INSERT INTO pipeline_alerts (alert_key, message, first_seen_utc) VALUES (?, ?, ?)",
                         (key, message, report.now))

    summary = {"fetched": report.fetched, "predicted": report.predicted, "blocked": report.blocked,
               "scored": report.scored, "new_alerts": report.new_alerts}
    with conn:
        # The one UPDATE in the pipeline's own bookkeeping (pipeline_ticks is
        # a run record, not a prediction, so it isn't append-only).
        conn.execute("UPDATE pipeline_ticks SET finished_at_utc = ?, report_json = ?, new_alerts = ? "
                     "WHERE tick_id = ?", (to_utc_text(datetime.now(timezone.utc)), json.dumps(summary),
                                           len(report.new_alerts), tick_id))
    for line in report.lines():
        logger.info(line)
    return report
