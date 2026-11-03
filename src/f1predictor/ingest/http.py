"""A small HTTP client for JSON APIs: polite rate limiting + retries with backoff.

Why not just call requests.get() directly?
    1. Rate limits: every API caps how fast we may call it (Jolpica: ~4/second
       and 200/hour; OpenF1: 3/second and 30/minute; see config.py). Firing
       requests in a tight loop gets us blocked (HTTP 429). So we enforce a
       minimum gap between requests.
    2. Flaky networks and servers: a request can fail for reasons that fix
       themselves (timeouts, a 503 while the server restarts). Retrying a few
       times, waiting longer each time, turns most of these into non-events.
       This is the PRD's "retry with backoff" requirement.

Everything that varies (how long to wait, how to sleep, which HTTP session)
is a constructor argument. That lets tests swap in fakes and run instantly,
without real network calls or real sleeping.
"""

import logging
import time
from typing import Any, Callable

import requests

# A logger named after this module ("f1predictor.ingest.http"). Using logging
# instead of print() means scripts decide how much detail to show, and messages
# carry timestamps and the module they came from.
logger = logging.getLogger(__name__)

# HTTP status codes worth retrying: the problem is likely temporary.
#   429 = Too Many Requests (we're being rate limited, so slow down)
#   500/502/503/504 = server-side errors (crash, bad gateway, overloaded, timeout)
# Anything else (e.g. 404 Not Found, 400 Bad Request) means the request itself
# is wrong, and retrying the same request would just fail again.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class FetchError(Exception):
    """Raised when a request still fails after all retries."""


class JsonHttpClient:
    """Makes GET requests that return JSON, with rate limiting and retries."""

    def __init__(
        self,
        min_interval_s: float = 0.5,
        max_retries: int = 5,
        backoff_base_s: float = 2.0,
        timeout_s: float = 30.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            min_interval_s: Minimum seconds between the START of two requests.
                0.5s = at most 2 requests/second, safely under Jolpica's burst limit.
            max_retries: How many times to retry after the first attempt fails.
            backoff_base_s: First retry waits this long; each retry doubles it
                (2s, 4s, 8s, 16s, 32s). Doubling ("exponential backoff") gives
                a struggling server progressively more room to recover.
            timeout_s: Give up on a single request if no response in this long.
                Without a timeout, a hung connection would block forever.
            session: A requests.Session reuses the underlying connection across
                requests (faster). Tests pass a fake object here instead.
            sleep / clock: Functions for waiting and telling time. Real ones by
                default; tests pass fakes so nothing actually waits.
        """
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.timeout_s = timeout_s
        self.session = session if session is not None else requests.Session()
        self._sleep = sleep
        self._clock = clock
        # When the previous request started. None = no request made yet.
        self._last_request_at: float | None = None

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """GET a URL and return the parsed JSON body.

        The body is usually a dict (Jolpica) or a list (OpenF1), hence `Any`.

        Retries temporary failures with exponential backoff, and raises
        FetchError if every attempt fails. Raises immediately (no retry) for
        non-retryable errors like 404.
        """
        # range(max_retries + 1): the first attempt plus max_retries retries.
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()

            # How long to wait before the next attempt if this one fails.
            # Exponential: attempt 0 -> 2s, attempt 1 -> 4s, attempt 2 -> 8s...
            retry_delay = self.backoff_base_s * (2 ** attempt)

            try:
                response = self.session.get(url, params=params, timeout=self.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as exc:
                # Network-level failure (DNS, dropped connection, timeout):
                # no response at all. Worth retrying.
                problem = f"network error: {exc}"
            else:
                # The `else` block of try/except runs only if no exception was raised.
                if response.status_code == 200:
                    return response.json()

                if response.status_code not in RETRYABLE_STATUS_CODES:
                    # A permanent error. raise_for_status() raises
                    # requests.HTTPError with the status code and URL.
                    response.raise_for_status()

                problem = f"HTTP {response.status_code}"

                # Servers that rate limit us often say exactly how long to wait
                # in a Retry-After header (in seconds). If present, respect it.
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None and retry_after.isdigit():
                    retry_delay = max(retry_delay, float(retry_after))

            # If we get here, this attempt failed with a retryable problem.
            if attempt == self.max_retries:
                # Out of retries. Raise a clear error instead of returning
                # nothing, so the caller can log it and move on.
                raise FetchError(
                    f"Giving up on {url} after {attempt + 1} attempts ({problem})"
                )

            logger.warning(
                "Attempt %d for %s failed (%s); retrying in %.0fs",
                attempt + 1, url, problem, retry_delay,
            )
            self._sleep(retry_delay)

        # Unreachable (the loop either returns or raises), but it tells type
        # checkers and readers that this function never falls off the end.
        raise AssertionError("unreachable")

    def _wait_for_rate_limit(self) -> None:
        """Sleep just long enough to keep min_interval_s between requests."""
        now = self._clock()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            if elapsed < self.min_interval_s:
                self._sleep(self.min_interval_s - elapsed)
        # Record when THIS request starts (after any sleeping).
        self._last_request_at = self._clock()
