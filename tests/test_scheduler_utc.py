"""The scheduler and time zones (test-suite step 8).

Two things the pipeline depends on that no other test looks at:

1. THE WORKFLOW FILE (.github/workflows/tick.yml). It's what runs the tick
   every hour; nothing in Python calls it, so a careless edit (dropping the
   "Run workflow" button, changing the cron, letting two runs overlap) would
   pass every other test and only show up as a missed prediction. These
   tests read the file and pin the parts that matter.

2. TIME ZONES. Every time in the pipeline is UTC: GitHub's cron, the
   database ('...Z'), the planner. Times arrive in other forms: OpenF1 sends
   '+00:00', a person types `--now` on a laptop in Toronto, a test passes
   a Los Angeles time. The rule (pipeline/plan.as_utc / parse_utc): any
   explicit timezone is converted to UTC; a time with NO timezone is
   refused, never guessed (Python would silently read it as the machine's
   local time, which is different on a laptop and on GitHub's runner).

No network: the planner tests use the fake world's calendar (tests/fake_f1.py).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from f1predictor.db.connection import get_connection, init_db
from f1predictor.ingest import jolpica
from f1predictor.ingest.openf1 import to_utc_iso
from f1predictor.pipeline.plan import LAST_CALL, as_utc, parse_utc, plan, to_utc_text
from f1predictor.pipeline.tick import run_tick
from f1predictor.predict import stage

from fake_f1 import FakeF1

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "tick.yml"
UTC = timezone.utc
TORONTO = ZoneInfo("America/Toronto")
LOS_ANGELES = ZoneInfo("America/Los_Angeles")
TOKYO = ZoneInfo("Asia/Tokyo")


# -----------------------------------------------------------------------------
# 1. The workflow file
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def workflow():
    data = yaml.safe_load(WORKFLOW.read_text())
    # YAML 1.1 reads the bare key `on:` as the boolean True, so PyYAML gives
    # {True: {...}} instead of {"on": {...}}. Accept either.
    data["on"] = data.pop(True, data.get("on"))
    return data


def tick_step(workflow):
    return next(s for s in workflow["jobs"]["tick"]["steps"] if s.get("id") == "tick")


def test_runs_every_hour_at_7_past(workflow):
    # One schedule, hourly, at :07 (GitHub's cron is always UTC; :00 is the
    # busiest minute, when scheduled runs start late or get dropped).
    assert workflow["on"]["schedule"] == [{"cron": "7 * * * *"}]


def test_a_late_or_skipped_hourly_run_still_leaves_time_before_each_cutoff():
    # GitHub may start a scheduled run late, or skip one when it's busy.
    # A stage's last call (log it even if data is missing) opens LAST_CALL
    # before its cutoff; with runs an hour apart, a last call longer than an
    # hour means at least two runs fall inside it, so one lost run can't
    # cost a prediction. Shortening LAST_CALL to an hour or less breaks that.
    assert LAST_CALL > timedelta(hours=1)


def test_the_run_workflow_button_exists_with_a_dry_run_option(workflow):
    # workflow_dispatch = the "Run workflow" button in GitHub's Actions tab:
    # how a person runs a tick by hand (e.g. to check something after a fix)
    # without waiting for the hour. Its one input: a dry run, off by default.
    dispatch = workflow["on"]["workflow_dispatch"]
    assert dispatch["inputs"]["dry_run"]["type"] == "boolean"
    assert dispatch["inputs"]["dry_run"]["default"] is False


def test_dry_run_reaches_the_script_only_when_asked_for(workflow):
    # The tick step turns the input into the --dry-run flag. A scheduled run
    # has no inputs at all, so `inputs.dry_run` is empty and the flag must
    # be left off: the hourly runs are the live ones.
    command = tick_step(workflow)["run"]
    assert "python scripts/tick.py ${{ inputs.dry_run && '--dry-run' || '' }}" in command
    # And never a pretend time: --now is for dry runs on a laptop only.
    assert "--now" not in command


def test_two_runs_never_overlap_and_none_is_cancelled(workflow):
    # Each run downloads the database, changes it and uploads it. Two at
    # once would lose one's work, so they queue (one group), and a queued
    # run waits rather than cancelling the one in progress (which could be
    # halfway through writing a prediction).
    assert workflow["concurrency"] == {"group": "pipeline-database", "cancel-in-progress": False}


def test_the_pipeline_switch_and_a_time_limit(workflow):
    job = workflow["jobs"]["tick"]
    # Off unless the repository variable says so (how it's paused).
    assert job["if"] == "vars.PIPELINE_ENABLED == 'true'"
    # A hung run must not hold the queue for GitHub's default 6 hours (every
    # later run waits behind it, and cutoffs pass): 30 minutes, then killed.
    assert job["timeout-minutes"] == 30


def test_a_new_alert_fails_the_run_but_only_after_the_upload(workflow):
    # GitHub emails when a run fails; that's the alert. But the database
    # must be uploaded first, or the tick's work (and the record of the
    # alert, so it isn't sent again) would be lost.
    names = [s.get("name") for s in workflow["jobs"]["tick"]["steps"]]
    assert names.index("Tick") < names.index("Upload the database") < names.index(
        "Fail if the tick raised a new alert")
    assert "set +e" in tick_step(workflow)["run"]     # the Tick step itself doesn't stop the job


def test_the_dashboard_is_exported_after_the_database_is_safe(workflow):
    # The export runs after the database upload (a failing export can't
    # cost the tick's work) and before the alert step (whose failure would
    # stop every later step).
    names = [s.get("name") for s in workflow["jobs"]["tick"]["steps"]]
    assert names.index("Upload the database") < names.index("Export the dashboard data") \
        < names.index("Set up Node") < names.index("Build the dashboard app") \
        < names.index("Package the site for GitHub Pages") < names.index("Fail if the tick raised a new alert")
    # The app is built (from the locked versions) and packaged only when
    # there's something new to deploy; most hours, nothing is built.
    steps = {s.get("name"): s for s in workflow["jobs"]["tick"]["steps"]}
    for name in ("Set up Node", "Build the dashboard app", "Package the site for GitHub Pages"):
        assert steps[name]["if"] == "steps.export.outputs.changed == 'true'"
    assert "npm ci" in steps["Build the dashboard app"]["run"]
    assert "cp -r dist/. ../site/" in steps["Build the dashboard app"]["run"]
    export = next(s for s in workflow["jobs"]["tick"]["steps"] if s.get("name") == "Export the dashboard data")
    assert "python scripts/export_dashboard.py data/f1.sqlite site" in export["run"]
    # Deploy only when something changed: the data's hash or the code's commit.
    assert 'hash\'])")@${GITHUB_SHA}"' in export["run"]


def test_pages_deploys_only_when_changed_even_if_the_tick_alerted(workflow):
    pages = workflow["jobs"]["pages"]
    assert pages["needs"] == "tick"
    # always(): the tick job "fails" on purpose to email an alert; the site
    # must still be updated then. But only when there's something new.
    assert pages["if"] == "always() && needs.tick.outputs.deploy == 'true'"
    assert workflow["jobs"]["tick"]["outputs"]["deploy"] == "${{ steps.export.outputs.changed }}"
    # The narrow permissions Pages needs, only in this job.
    assert pages["permissions"] == {"pages": "write", "id-token": "write"}
    assert pages["environment"]["name"] == "github-pages"
    assert [s["uses"] for s in pages["steps"]] == ["actions/deploy-pages@v4"]


# -----------------------------------------------------------------------------
# 2. Times: one instant, any timezone; no timezone, refused
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "2026-11-01T20:00:00Z",              # the database's form, and Jolpica's
    "2026-11-01T20:00:00+00:00",         # OpenF1's form
    "2026-11-01T14:00:00-06:00",         # Mexico City local time
    "2026-11-02T05:00:00+09:00",         # Tokyo, already the next day
])
def test_the_same_instant_written_any_way_is_the_same_utc_time(text):
    assert to_utc_text(parse_utc(text)) == "2026-11-01T20:00:00Z"


def test_a_time_without_a_timezone_is_refused_not_guessed():
    # The danger: `scripts/tick.py --dry-run --now 2026-11-07T17:20:00` typed
    # on a laptop in Toronto used to mean 22:20 UTC (Python assumes local
    # time), five hours off, silently. Now it's an error that says why.
    with pytest.raises(ValueError, match="without a timezone"):
        parse_utc("2026-11-07T17:20:00")
    with pytest.raises(ValueError, match="without a timezone"):
        as_utc(datetime(2026, 11, 7, 17, 20))


def test_stored_text_has_one_format_whatever_the_source():
    # The database compares times as TEXT (e.g. "races starting between
    # these two times", pipeline/plan._events_near). That's only correct if
    # every stored time has exactly the same shape. OpenF1 (with an offset)
    # and Jolpica (date + time) must come out identical for the same instant.
    from_openf1 = to_utc_iso("2026-11-01T14:00:00-06:00", timespec="seconds")
    from_jolpica = jolpica.to_utc_timestamp("2026-11-01", "20:00:00Z")
    from_pipeline = to_utc_text(datetime(2026, 11, 1, 14, 0, tzinfo=ZoneInfo("America/Mexico_City")))
    assert from_openf1 == from_jolpica == from_pipeline == "2026-11-01T20:00:00Z"


# -----------------------------------------------------------------------------
# 3. The planner doesn't care how `now` is written
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calendar_db():
    """The fake 2026 + 2027 seasons from Jolpica (schedule and the 2026
    results): enough for the planner to plan fetches and predictions."""
    conn = get_connection(":memory:")
    init_db(conn)
    world = FakeF1(now=datetime(2027, 3, 1, tzinfo=UTC))
    for season in (2026, 2027):
        jolpica.ingest_season(conn, world, season)
    return conn


def test_the_plan_is_the_same_in_every_timezone_hour_by_hour(calendar_db):
    # Every hour from the Monday before round 1 to the Tuesday after round 2,
    # the plan must be identical whether `now` is given in UTC, Los Angeles
    # or Tokyo time. Round 2 (2027-03-14) is also the day the US moves its
    # clocks forward, so Los Angeles's offset changes mid-weekend (-8 to -7):
    # a local-time mistake would show up as a plan an hour off.
    now, end = datetime(2027, 3, 1, tzinfo=UTC), datetime(2027, 3, 16, 12, tzinfo=UTC)
    busy_hours = 0
    while now <= end:
        expected = plan(calendar_db, now)
        for zone in (LOS_ANGELES, TOKYO):
            assert plan(calendar_db, now.astimezone(zone)) == expected, f"{now} in {zone}"
        busy_hours += bool(expected)
        now += timedelta(hours=1)
    # (Control: the plans weren't all empty, so the comparison meant something.)
    assert busy_hours > 100


def test_the_season_is_the_utc_year_on_new_years_eve():
    # 23:00 UTC on Dec 31 is already Jan 1 in Tokyo. The planner fetches a
    # season's calendar when "this year" has none; "this year" must be
    # the UTC year, or the first January ticks would plan the wrong season.
    conn = get_connection(":memory:")
    init_db(conn)
    late = datetime(2026, 12, 31, 23, 0, tzinfo=UTC)
    targets = {(a.kind, a.season) for a in plan(conn, late.astimezone(TOKYO))}
    assert targets == {("fetch_jolpica", 2026)}
    assert {(a.kind, a.season) for a in plan(conn, late + timedelta(hours=1))} == {("fetch_jolpica", 2027)}


def test_the_planner_and_the_tick_refuse_a_time_without_a_timezone(calendar_db):
    with pytest.raises(ValueError, match="without a timezone"):
        plan(calendar_db, datetime(2027, 3, 5, 12, 0))
    with pytest.raises(ValueError, match="without a timezone"):
        run_tick(calendar_db, now=datetime(2027, 3, 5, 12, 0), jolpica_client=object(), openf1_client=object())


def test_a_tick_given_a_local_time_records_and_does_exactly_what_a_utc_tick_does(monkeypatch):
    # One whole tick, twice, on two fresh databases: once at a UTC time and
    # once at the same instant written as Toronto time. Same fetches, same
    # prediction, and the tick is recorded under the UTC time.
    monkeypatch.setattr(stage, "model_version", lambda: "abc1234+settings.00000000")
    instant = datetime(2027, 3, 5, 0, 7, tzinfo=UTC)     # Friday of round 1: pre-weekend is due

    def one_tick(now):
        conn = get_connection(":memory:")
        init_db(conn)
        world = FakeF1(now=instant)
        jolpica.ingest_season(conn, world, 2026)
        report = run_tick(conn, now=now, jolpica_client=world, openf1_client=world, n_sims=500)
        as_of = conn.execute("SELECT as_of_utc FROM prediction_runs").fetchall()
        return report, [r[0] for r in as_of]

    utc_report, utc_runs = one_tick(instant)
    local_report, local_runs = one_tick(instant.astimezone(TORONTO))
    assert local_report.now == utc_report.now == "2027-03-05T00:07:00Z"
    assert local_report.fetched == utc_report.fetched
    assert [p.split(" -> ")[0] for p in local_report.predicted] == ["2027 R1 pre_weekend"]
    assert local_report.predicted == utc_report.predicted
    assert local_runs == utc_runs == ["2027-03-05T12:00:00Z"]    # FP1, in UTC
