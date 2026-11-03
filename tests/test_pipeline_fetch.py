"""Tests for the pipeline's fetch step (pipeline/fetch.py).

The loaders themselves are tested in test_openf1.py / test_jolpica.py. Here
the question is routing: does each planned fetch reach the right loader with
the right filters, exactly once?
"""

from datetime import datetime, timezone

from f1predictor.pipeline import fetch
from f1predictor.pipeline.plan import Action

NOW = datetime(2027, 3, 14, 18, 0, tzinfo=timezone.utc)


def action(kind, target, rnd=2, season=2027):
    return Action(kind, event_id=1, season=season, round=rnd, target=target, reason="test")


def test_fetches_are_routed_to_the_loaders(monkeypatch):
    calls = []
    monkeypatch.setattr(fetch.openf1, "ingest_season",
                        lambda conn, client, season, **kw: calls.append(("openf1", season, kw)) or {"o": "ok"})
    monkeypatch.setattr(fetch.jolpica, "ingest_season",
                        lambda conn, client, season: calls.append(("jolpica", season)) or {"j": "ok"})
    monkeypatch.setattr(fetch, "rematch_unmatched_entries", lambda conn: 0)

    actions = [action("fetch_session", "R"), action("fetch_session", "grid"),
               action("fetch_jolpica", "qualifying"), action("fetch_jolpica", "results"),
               action("predict", "post_quali")]                         # not a fetch: ignored
    outcomes = fetch.run_fetches(None, actions, NOW, jolpica_client=None, openf1_client=None)

    assert outcomes == {"o": "ok", "j": "ok"}
    openf1_calls = [c for c in calls if c[0] == "openf1"]
    assert len(openf1_calls) == 1                                       # one call for the round
    kwargs = openf1_calls[0][2]
    assert kwargs["rounds"] == {2} and kwargs["session_types"] == {"R", "Q"}   # grid -> Q
    assert kwargs["settle_hours"] == 0 and kwargs["now"] == NOW
    assert [c for c in calls if c[0] == "jolpica"] == [("jolpica", 2027)]      # once per season


def test_nothing_to_fetch_makes_no_calls(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("no loader should be called")
    monkeypatch.setattr(fetch.openf1, "ingest_season", fail)
    monkeypatch.setattr(fetch.jolpica, "ingest_season", fail)
    monkeypatch.setattr(fetch, "rematch_unmatched_entries", fail)
    assert fetch.run_fetches(None, [action("score", "race")], NOW, None, None) == {}
