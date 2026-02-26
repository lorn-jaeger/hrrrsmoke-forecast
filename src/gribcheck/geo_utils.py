from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from math import asin, cos, radians, sin, sqrt

import numpy as np
from pyproj import CRS, Transformer


EARTH_RADIUS_KM = 6371.0088

# HRRR native Lambert conformal projection.
HRRR_CRS = CRS.from_proj4(
    "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 "
    "+a=6371229 +b=6371229 +units=m +no_defs"
)
WGS84 = CRS.from_epsg(4326)


def hrrr_transformer_to_xy() -> Transformer:
    return Transformer.from_crs(WGS84, HRRR_CRS, always_xy=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r, lon1r, lat2r, lon2r = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = sin(dlat / 2) ** 2 + cos(lat1r) * cos(lat2r) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def min_distance_to_bbox_km(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> float:
    min_lon, min_lat, max_lon, max_lat = bbox
    clamped_lon = min(max(lon, min_lon), max_lon)
    clamped_lat = min(max(lat, min_lat), max_lat)
    return haversine_km(lat, lon, clamped_lat, clamped_lon)


def bbox_radius_km(bbox: tuple[float, float, float, float]) -> float:
    min_lon, min_lat, max_lon, max_lat = bbox
    diag = haversine_km(min_lat, min_lon, max_lat, max_lon)
    return diag / 2.0


def nested_coordinates_bounds(coords: Iterable) -> tuple[float, float, float, float]:
    min_lon = float("inf")
    min_lat = float("inf")
    max_lon = float("-inf")
    max_lat = float("-inf")

    stack = [coords]
    while stack:
        current = stack.pop()
        if not isinstance(current, (list, tuple)) or not current:
            continue
        if isinstance(current[0], (float, int, Decimal)) and len(current) >= 2:
            lon = float(current[0])
            lat = float(current[1])
            min_lon = min(min_lon, lon)
            min_lat = min(min_lat, lat)
            max_lon = max(max_lon, lon)
            max_lat = max(max_lat, lat)
        else:
            stack.extend(current)

    if min_lon == float("inf"):
        raise ValueError("Unable to compute bounds from coordinates")

    return (min_lon, min_lat, max_lon, max_lat)


def split_for_date(value, split_config) -> str:
    if split_config.train_start <= value <= split_config.train_end:
        return "train"
    if split_config.val_start <= value <= split_config.val_end:
        return "val"
    if split_config.test_start <= value <= split_config.test_end:
        return "test"
    return "outside"


def to_numpy_float16(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float16, copy=False)
