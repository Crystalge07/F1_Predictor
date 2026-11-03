"""Tests for the retrying HTTP client, using fakes instead of the real network.

The client accepts `session`, `sleep` and `clock` as arguments, so here we pass
fake versions: a session that returns scripted responses, and a sleep that only
records how long it WOULD have waited. Tests run instantly and offline.
"""

import pytest
import requests

from f1predictor.ingest.http import FetchError, JsonHttpClient


class FakeResponse:
    """Just enough of requests.Response for our client to use."""

    def __init__(self, status_code: int, body: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self._body = body or {}
        self.headers = headers or {}

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Returns the scripted responses (or raises scripted errors) in order."""

    def __init__(self, outcomes: list):
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(outcomes: list, max_retries: int = 3):
    """Build a client wired to fakes. Returns (client, session, sleeps list)."""
    session = FakeSession(outcomes)
    sleeps: list[float] = []
    client = JsonHttpClient(
        min_interval_s=0,           # no rate-limit waits, so `sleeps` only holds backoffs
        max_retries=max_retries,
        backoff_base_s=2.0,
        session=session,
        sleep=sleeps.append,        # "sleeping" just records the duration
        clock=lambda: 0.0,
    )
    return client, session, sleeps


def test_returns_json_on_success():
    client, session, sleeps = make_client([FakeResponse(200, {"ok": True})])
    assert client.get_json("http://x") == {"ok": True}
    assert session.calls == 1
    assert sleeps == []


def test_retries_server_errors_with_exponential_backoff():
    client, session, sleeps = make_client([
        FakeResponse(503),
        requests.ConnectionError("dropped"),
        FakeResponse(200, {"ok": True}),
    ])
    assert client.get_json("http://x") == {"ok": True}
    assert session.calls == 3
    assert sleeps == [2.0, 4.0]     # doubled after each failure


def test_does_not_retry_404():
    client, session, sleeps = make_client([FakeResponse(404)])
    with pytest.raises(requests.HTTPError):
        client.get_json("http://x")
    assert session.calls == 1       # gave up immediately


def test_gives_up_after_max_retries():
    client, session, sleeps = make_client([FakeResponse(500)] * 3, max_retries=2)
    with pytest.raises(FetchError):
        client.get_json("http://x")
    assert session.calls == 3       # first attempt + 2 retries


def test_respects_retry_after_header():
    client, session, sleeps = make_client([
        FakeResponse(429, headers={"Retry-After": "30"}),
        FakeResponse(200, {"ok": True}),
    ])
    client.get_json("http://x")
    assert sleeps == [30.0]         # server said 30s, which beats our 2s backoff


def test_rate_limit_spaces_out_requests():
    # A clock that reads 0.0, then 0.0, then 0.1: the second request starts
    # only 0.1s after the first, so the client should wait the remaining 0.9s.
    times = iter([0.0, 0.0, 0.1, 1.0])
    sleeps: list[float] = []
    client = JsonHttpClient(
        min_interval_s=1.0,
        session=FakeSession([FakeResponse(200), FakeResponse(200)]),
        sleep=sleeps.append,
        clock=lambda: next(times),
    )
    client.get_json("http://x")
    client.get_json("http://x")
    assert sleeps == [pytest.approx(0.9)]
