from datetime import date

from gribcheck.pipelines.station_daily import _expected_hours_for_local_day


def test_expected_hours_for_dst_transitions():
    # DST spring forward in America/Los_Angeles: 23-hour day.
    assert _expected_hours_for_local_day(date(2021, 3, 14), "America/Los_Angeles") == 23

    # DST fall back: 25-hour day.
    assert _expected_hours_for_local_day(date(2021, 11, 7), "America/Los_Angeles") == 25
