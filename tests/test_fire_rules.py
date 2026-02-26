from datetime import date

from gribcheck.fire import _choose_fire_end, _feature_to_fire_record


def test_fire_end_fallback_priority():
    props = {
        "attr_FireOutDateTime": "",
        "attr_ContainmentDateTime": "",
        "attr_ControlDateTime": "",
        "poly_PolygonDateTime": "Fri, 13 Feb 2026 14:30:58 GMT",
    }
    dt = _choose_fire_end(props)
    assert dt is not None
    assert dt.year == 2026


def test_feature_to_fire_record_filters_and_fields():
    feature = {
        "properties": {
            "attr_IncidentTypeCategory": "WF",
            "attr_FireDiscoveryDateTime": "Fri, 01 Jul 2022 00:00:00 GMT",
            "attr_FireOutDateTime": "Sat, 02 Jul 2022 00:00:00 GMT",
            "attr_IncidentSize": "70",
            "attr_UniqueFireIdentifier": "abc-1",
            "attr_IncidentName": "Test Fire",
            "attr_POOState": "US-CA",
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-120.0, 38.0], [-119.8, 38.0], [-119.8, 38.2], [-120.0, 38.2], [-120.0, 38.0]]],
        },
    }

    rec = _feature_to_fire_record(
        feature=feature,
        min_size_acres=50.0,
        run_start=date(2021, 1, 1),
        run_end=date(2025, 10, 31),
        incident_type="WF",
    )
    assert rec is not None
    assert rec.unique_fire_id == "abc-1"
    assert rec.incident_name == "Test Fire"
