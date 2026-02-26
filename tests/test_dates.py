from datetime import date

from gribcheck.date_utils import parse_optional_datetime, parse_pm_date


def test_parse_pm_date_mixed_formats():
    assert parse_pm_date("2021-07-04") == date(2021, 7, 4)
    assert parse_pm_date("7/4/2021") == date(2021, 7, 4)


def test_parse_optional_datetime_formats():
    dt_rfc = parse_optional_datetime("Sat, 19 Nov 2022 00:52:20 GMT")
    assert dt_rfc is not None
    assert dt_rfc.year == 2022
    assert dt_rfc.month == 11

    dt_us = parse_optional_datetime("2/12/2026 4:44:00 PM")
    assert dt_us is not None
    assert dt_us.year == 2026
    assert dt_us.hour == 16
