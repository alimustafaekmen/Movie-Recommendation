"""Turkish and English: the message catalogue and the date, number and runtime formats.

Every text the pages show is a key of movierec/messages.json, which holds it in both
languages with {name} placeholders. The functions here take the language, so they need no
Flask.
"""

import datetime
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# English first: Werkzeug's best_match gives a wildcard (Accept-Language: *) or a tie to the
# first language, and a browser that asks for neither language gets English.
LANGUAGES = ("en", "tr")
DEFAULT_LANGUAGE = "en"
# The language parameter of TMDb requests.
TMDB_LANGUAGES = {"tr": "tr-TR", "en": "en-US"}
MESSAGES = json.loads(Path(__file__).with_name("messages.json").read_text(encoding="utf-8"))

_MONTHS = {
    "tr": (
        "Ocak",
        "Şubat",
        "Mart",
        "Nisan",
        "Mayıs",
        "Haziran",
        "Temmuz",
        "Ağustos",
        "Eylül",
        "Ekim",
        "Kasım",
        "Aralık",
    ),
    "en": (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ),
}
# "05 Mart 1980" and "March 5, 1980".
_DATE_PATTERNS = {"tr": "{day:02d} {month} {year}", "en": "{month} {day}, {year}"}
# The thousands separator and the decimal mark.
_MARKS = {"tr": (".", ","), "en": (",", ".")}


def translate(lang: str, key: str, **params) -> str:
    """Return the text of a catalogue key in lang, with its {name} placeholders filled in.

    An unknown key or a missing parameter raises KeyError, so a typo fails the page.
    """
    return MESSAGES[key][lang].format(**params)


def translate_count(lang: str, key: str, count: int, **params) -> str:
    """Return the text for count: key.one for 1, key.other otherwise; {count} is formatted."""
    form = "one" if count == 1 else "other"
    return translate(lang, f"{key}.{form}", count=format_number(count, lang), **params)


def message(key: str, **params) -> dict:
    """Return a message for the web layer to translate: its catalogue key and parameters."""
    return {"key": key, "params": params}


def format_date(value: str | None, lang: str) -> str:
    """Format an ISO date such as '1980-03-05' as '05 Mart 1980' or 'March 5, 1980'."""
    if not value:
        return translate(lang, "date.unknown")
    date_text = str(value)
    if not date_text.strip():
        return translate(lang, "date.unknown")
    try:
        day = datetime.date.fromisoformat(date_text)
    except ValueError:
        return date_text
    month_name = _MONTHS[lang][day.month - 1]
    return _DATE_PATTERNS[lang].format(day=day.day, month=month_name, year=day.year)


def format_number(value: int | float | None, lang: str) -> str:
    """Format a count such as 12002 as '12.002' or '12,002'."""
    if value is None:
        return ""
    thousands, _ = _MARKS[lang]
    # The "," format option groups thousands with commas (12,002).
    with_commas = f"{int(value):,}"
    return with_commas.replace(",", thousands)


def format_rating(value: float | None, lang: str) -> str:
    """Format a rating such as 8.2 as '8,2' or '8.2'; no rating (None or 0) gives ''."""
    if not value:
        return ""
    # Round half up from the shortest decimal text ("8.25"), as toLocaleString does in the
    # browser; round() rounds halves to even: round(8.25, 1) is 8.2.
    shortest_text = str(float(value))
    rating = Decimal(shortest_text).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return decimal_text(str(rating), lang)


def decimal_text(text: str, lang: str) -> str:
    """Write a number given as text with a decimal point, such as '7.5', as lang writes it."""
    _, decimal_mark = _MARKS[lang]
    return text.replace(".", decimal_mark)


def format_runtime(minutes, lang: str) -> str | None:
    """Format a runtime in minutes, e.g. 152 -> '2 sa 32 dk' or '2 h 32 min'."""
    try:
        value = float(minutes)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    total_minutes = int(value)
    if total_minutes <= 0:
        return None

    hours, minutes_left = divmod(total_minutes, 60)  # 152 -> 2 h, 32 min
    if hours and minutes_left:
        return translate(lang, "runtime.hours_minutes", hours=hours, minutes=minutes_left)
    if hours:
        return translate(lang, "runtime.hours", hours=hours)
    return translate(lang, "runtime.minutes", minutes=minutes_left)
