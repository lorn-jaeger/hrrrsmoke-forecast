from datetime import date

import pandas as pd

from gribcheck.pipelines.pm_ingest import _combine_primary_fallback, _dedupe_within_code


def test_dedupe_within_code_ranking():
    df = pd.DataFrame(
        [
            {
                "station_id": "01-001-0001",
                "date_local": date(2021, 1, 1),
                "observation_percent": 90.0,
                "poc_numeric": 2,
                "date_last_change": pd.Timestamp("2024-01-01"),
                "poc": "2",
                "method_name": "M2",
                "parameter_name": "PM",
                "pm_source_code": "88101",
            },
            {
                "station_id": "01-001-0001",
                "date_local": date(2021, 1, 1),
                "observation_percent": 95.0,
                "poc_numeric": 5,
                "date_last_change": pd.Timestamp("2024-01-02"),
                "poc": "5",
                "method_name": "M5",
                "parameter_name": "PM",
                "pm_source_code": "88101",
            },
            {
                "station_id": "01-001-0001",
                "date_local": date(2021, 1, 1),
                "observation_percent": 95.0,
                "poc_numeric": 1,
                "date_last_change": pd.Timestamp("2024-01-01"),
                "poc": "1",
                "method_name": "M1",
                "parameter_name": "PM",
                "pm_source_code": "88101",
            },
        ]
    )

    out = _dedupe_within_code(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["poc"] == "1"


def test_primary_precedence_on_overlap():
    common = {
        "state_code": "01",
        "county_code": "001",
        "site_num": "0001",
        "station_id": "01-001-0001",
        "date_local": date(2021, 1, 1),
        "latitude": 30.0,
        "longitude": -90.0,
        "arithmetic_mean": 12.0,
        "units": "ug/m3",
        "observation_count": 1,
        "observation_percent": 100.0,
        "aqi": 45,
        "event_type": "None",
        "state_name": "Alabama",
        "county_name": "X",
        "city_name": "Y",
        "cbsa_name": "Z",
        "pm_source_poc": "1",
        "pm_source_method": "M",
        "pm_source_parameter_name": "PM",
        "pollutant_standard": "std",
        "sample_duration": "24 HOUR",
        "date_last_change": pd.Timestamp("2024-01-01"),
        "poc_numeric": 1,
    }

    primary = pd.DataFrame([{**common, "pm_source_code": "88101"}])
    fallback = pd.DataFrame([{**common, "pm_source_code": "88502", "arithmetic_mean": 99.0}])

    out = _combine_primary_fallback(primary, fallback)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["pm_source_code"] == "88101"
    assert float(row["pm25_value"]) == 12.0
