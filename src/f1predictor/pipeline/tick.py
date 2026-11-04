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
    3. SCORE    plan again, then score every race whose result is in, and
                re-score any whose official result changed since (a late
                penalty: plan.RESULT_RECHECKS re-fetches results 24h and
                72h after each race; a change raises 'result_revised'). Then
                the guardrail (pipeline/drift.py): alert if, over the last 8
                scored live races, the model is >15% worse than its
                baseline on places off or podium Brier. Alert only.
    4. REPORT   record the tick in pipeline_ticks, and raise alerts for
                problems that matter: data that failed validation (bad,
                not just late: alerted at once), blocked predictions
                (including a run that failed to write), scoring that
                failed, stages missed, sessions given up on, Jolpica data
                late. (A fetch that failed because the data isn't there yet
                isn't one: data a few minutes late is normal.) Each alert is raised once
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

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from f1predictor.evaluate.scorer import (record_scored_snapshot, result_changes, result_snapshot,
                                         score_runs, scored_snapshot)
from f1predictor.pipeline.drift import drift_alerts
from f1predictor.pipeline.fetch import run_fetches
from f1predictor.pipeline.plan import Action, as_utc, parse_utc, plan, to_utc_text
from f1predictor.predict.stage import live_problems, make_prediction, write_run

logger = logging.getLogger(__name__)

# The pipeline is meant to run every hour. A run that starts more than this
# long after the previous one raises a "gap" alert (decision 2026-11-04):
# a stage may have been missed while nothing ran. It can only be said once
# runs resume; the real-time warning is the outside dead-man's switch the
# workflow pings (healthchecks.io: .github/workflows/tick.yml).
MAX_GAP = timedelta(hours=3)


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

    # Any timezone is accepted and converted; a time with none is refused
    # (pipeline/plan.as_utc), so a tick can't silently run hours off.
    now = as_utc(now or datetime.now(timezone.utc))
    if jolpica_client is None or openf1_client is None:
        jolpica_client, openf1_client = make_clients()
    report = TickReport(now=to_utc_text(now), dry_run=dry_run)
    # When the previous run started, for the gap check below. Real clock
    # times on both sides (started_at_utc), not `now`, which a test or a
    # --now dry run can set to any time.
    previous = conn.execute("SELECT MAX(started_at_utc) FROM pipeline_ticks").fetchone()[0]
    started = datetime.now(timezone.utc)
    tick_id = conn.execute("INSERT INTO pipeline_ticks (started_at_utc, dry_run) VALUES (?, ?)",
                           (to_utc_text(started), int(dry_run))).lastrowid
    conn.commit()
    if previous is not None and started - parse_utc(previous) > MAX_GAP:
        hours = (started - parse_utc(previous)).total_seconds() / 3600
        # Keyed by the previous run: one alert per gap, however it ends.
        report.alerts.append((f"gap:{previous}",
                              f"gap: no pipeline run for {hours:.1f} hours (previous run started {previous}); "
                              "runs have resumed. Check that the hourly trigger (cron-job.org) and GitHub Actions "
                              "are working, and whether a stage was missed (a 'missed' alert says so)."))

    # --- 1. Fetch ------------------------------------------------------------
    # A failed fetch is NOT an alert by itself: data that's a few minutes
    # late is normal, and the next tick retries. It shows in the report;
    # problems that last are alerted by the planner ('gave_up', 'late'), and
    # what really matters (a prediction blocked or missed) is alerted below.
    report.fetched = run_fetches(conn, plan(conn, now), now, jolpica_client, openf1_client)
    # ...but BAD data is (decision 2026-11-03): a fetch that failed because
    # validation found errors (a partial result, a driver missing) is not
    # going to fix itself in a few minutes like late data does. The batch
    # was rolled back, so nothing of it is saved, and the loaders stored the
    # errors in validation_issues under the item's key. Alerted once per
    # item; the next ticks still retry, in case the source corrects it.
    for item_key, status in report.fetched.items():
        if status != "failed":
            continue
        errors = [row["detail"] for row in conn.execute(
            "SELECT detail FROM validation_issues WHERE item_key = ? AND severity = 'error' "
            "ORDER BY check_name", (item_key,))]
        if errors:
            report.alerts.append((f"invalid:{item_key}",
                                  f"invalid data, not saved: {item_key}: {'; '.join(errors)}; "
                                  "retried every tick until it validates"))

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
            continue
        try:
            # write_run is one transaction: if it fails (disk error, or the
            # database refusing a second live run, db/schema.sql), nothing
            # of the run is left behind, so the next tick can simply retry.
            run_id = write_run(conn, prediction, "live")
        except Exception as error:            # same rule: never crash the tick
            line = f"{_label(action)}: writing the run failed: {error}"
            report.blocked.append(line)
            report.alerts.append((f"blocked:{action.season}:{action.round}:{action.target}", f"not logged, {line}"))
            continue
        favourite = prediction.drivers.sort_values("p_win", ascending=False).iloc[0]
        report.predicted.append(f"{_label(action)} -> live run {run_id} "
                                f"(favourite {favourite.driver_id} {favourite.p_win:.0%})")

    # --- 3. Score ------------------------------------------------------------
    # 'score': runs not scored yet. 'rescore': every run of a race whose
    # official result changed after it was scored (plan.RESULT_RECHECKS);
    # scores are upserts, so re-scoring replaces them. Either way the result
    # the runs were scored against is remembered (scored_results), which is
    # how the planner notices the next change.
    for action in plan(conn, now):
        if action.kind not in ("score", "rescore") or dry_run:
            continue
        only_unscored = ("AND NOT EXISTS (SELECT 1 FROM scores s WHERE s.run_id = r.run_id)"
                         if action.kind == "score" else "")
        run_ids = [r[0] for r in conn.execute(
            f"SELECT run_id FROM prediction_runs r WHERE event_id = ? {only_unscored}", (action.event_id,))]
        scored_against = scored_snapshot(conn, action.event_id)
        snapshot = result_snapshot(conn, action.event_id)
        unresolved: dict[int, list[str]] = {}
        try:
            scored = score_runs(conn, run_ids, unresolved)
        except Exception as error:
            # Scoring must not crash the tick either (the alerts and the tick
            # record below still need to happen). Each run's scores are
            # written in their own transaction (evaluate/scorer.py), so runs
            # scored before the error keep their scores; the rest still have
            # none, so the planner plans 'score' again and a later tick
            # finishes them. Scores are upserts, so nothing is counted twice.
            report.alerts.append((f"score_failed:{action.season}:{action.round}",
                                  f"score_failed: {action.season} R{action.round}: {error}; retried next tick"))
            continue
        record_scored_snapshot(conn, action.event_id, snapshot)
        rescored = " re-scored" if action.kind == "rescore" else ""
        report.scored.append(f"{action.season} R{action.round}: {len(scored)} run(s){rescored}")
        # A result that changed after scoring is worth a person's look (the
        # scorecard just moved). Not alerted when there was simply no record
        # yet (races scored before scored_results existed). Keyed by the new
        # result, so each distinct revision alerts once.
        if action.kind == "rescore" and scored_against is not None:
            changes = result_changes(scored_against, snapshot)
            digest = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()[:8]
            report.alerts.append((f"result_revised:{action.season}:{action.round}:{digest}",
                                  f"result_revised: {action.season} R{action.round}: the official result changed "
                                  f"after scoring ({'; '.join(changes)}); {len(scored)} run(s) re-scored"))
        # A debutant predicted under a temporary id who can't be identified
        # (features/provisional.py): that run was left unscored, not scored
        # wrongly or dropped. Alerted once per car; every tick tries again
        # (e.g. once the car's match turns up), until the race is 10 days
        # old and leaves the planner's window (pipeline/plan.LOOK_BACK).
        for run_id, temporary_ids in unresolved.items():
            for temporary_id in temporary_ids:
                report.alerts.append((
                    f"unresolved_driver:{action.season}:{action.round}:{temporary_id}",
                    f"unresolved_driver: {action.season} R{action.round}: run {run_id} predicted "
                    f"{temporary_id} (a driver with no id yet), and nobody in the result can be "
                    "matched to that car, so the run is NOT scored. Check who drove it "
                    "(session_entries / results car numbers); the run is scored on the next "
                    "tick once the car matches one driver."))

    # --- 3b. Guardrail: is the live model falling behind its baseline? -----
    # Alert only (pipeline/drift.py). Runs every tick, but each alert names
    # the newest scored race, so it's raised once per race, not every hour.
    try:
        report.alerts.extend(drift_alerts(conn))
    except Exception as error:
        # Same rule as scoring: a broken check must not take the tick down.
        # Keyed by day, so a check that stays broken alerts once a day.
        report.alerts.append((f"drift_check_failed:{report.now[:10]}",
                              f"drift_check_failed: the guardrail check crashed: {error}"))

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
