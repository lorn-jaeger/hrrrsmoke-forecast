from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np

from gribcheck.models import FireRecord
from gribcheck.pipelines.wildfire_raster import _iter_fire_days, _mean_patches


def _record(start_day: date, end_day: date) -> FireRecord:
    return FireRecord(
        unique_fire_id="f1",
        incident_name="x",
        incident_type="WF",
        state="CA",
        start_time_utc=datetime(start_day.year, start_day.month, start_day.day, 0, tzinfo=timezone.utc),
        end_time_utc=datetime(end_day.year, end_day.month, end_day.day, 23, tzinfo=timezone.utc),
        start_date=start_day,
        end_date=end_day,
        size_acres=1000.0,
        min_lon=-121.0,
        min_lat=37.0,
        max_lon=-120.0,
        max_lat=38.0,
    )


def test_iter_fire_days_requires_next_day_label():
    rec = _record(date(2021, 1, 1), date(2021, 1, 3))

    days = list(
        _iter_fire_days(
            rec,
            cap=None,
            run_start=date(2021, 1, 1),
            run_end=date(2021, 1, 3),
            require_next_day=True,
        )
    )
    assert days == [date(2021, 1, 1), date(2021, 1, 2)]



def test_iter_fire_days_honors_caps_and_limits():
    rec = _record(date(2021, 1, 1), date(2021, 1, 5))

    days_cap = list(
        _iter_fire_days(
            rec,
            cap=2,
            run_start=date(2021, 1, 1),
            run_end=date(2021, 1, 5),
            require_next_day=False,
        )
    )
    assert days_cap == [date(2021, 1, 1), date(2021, 1, 2)]

    days_limit = list(
        _iter_fire_days(
            rec,
            cap=None,
            run_start=date(2021, 1, 1),
            run_end=date(2021, 1, 5),
            require_next_day=False,
            max_days_per_fire=1,
        )
    )
    assert days_limit == [date(2021, 1, 1)]



def test_mean_patches_computes_expected_average():
    p0 = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    p1 = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    out = _mean_patches([p0, p1])
    expected = np.array([[1.5, 3.5], [5.5, 7.5]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)
