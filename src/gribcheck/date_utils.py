from __future__ import annotations

from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime


def parse_pm_date(value: str) -> date:
    value = value.strip().strip('"')
    if not value:
        raise ValueError("Empty date")
    if "-" in value:
        return date.fromisoformat(value)
    month, day, year = value.split("/")
    return date(int(year), int(month), int(day))


def parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip().strip('"')
    if not text:
        return None

    # RFC 2822 style: Sat, 19 Nov 2022 00:52:20 GMT
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # M/D/YYYY H:MM:SS AM/PM
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue

    # ISO fallback
    try:
        dt = datetime.fromisoformat(text)
    except Exception as exc:  # pragma: no cover - debug branch
        raise ValueError(f"Unable to parse datetime: {text}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def season_from_date(value: date) -> str:
    month = value.month
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"
