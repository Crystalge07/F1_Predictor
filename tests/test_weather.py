"""Tests for race rain fractions."""

import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.features.weather import race_rain


@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    init_db(connection)
    connection.execute("INSERT INTO circuits (circuit_id, name) VALUES ('c', 'C')")
    connection.execute("INSERT INTO events (season, round, name, circuit_id, race_date) "
                       "VALUES (2024, 1, 'GP', 'c', '2024-03-01')")
    yield connection
    connection.close()


def test_rain_fraction_and_wet_flag(conn):
    race = conn.execute("INSERT INTO sessions (event_id, session_type) VALUES (1, 'R')").lastrowid
    quali = conn.execute("INSERT INTO sessions (event_id, session_type) VALUES (1, 'Q')").lastrowid
    # Race: 1 of 10 samples raining -> 0.1 (wet). Qualifying rain is ignored.
    for i in range(10):
        conn.execute("INSERT INTO weather (session_id, sample_time_utc, rainfall) VALUES (?, ?, ?)",
                     (race, f"2024-03-01T15:{i:02d}:00Z", int(i == 0)))
        conn.execute("INSERT INTO weather (session_id, sample_time_utc, rainfall) VALUES (?, ?, 1)",
                     (quali, f"2024-02-29T15:{i:02d}:00Z"))
    rain = race_rain(conn)
    assert rain["session_id"].tolist() == [race]
    assert rain["rain_fraction"].iloc[0] == pytest.approx(0.1)
    assert rain["wet"].iloc[0] == 1
