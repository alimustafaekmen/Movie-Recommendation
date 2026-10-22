"""Minimal TMDb v3 client: short timeouts, a cooldown after failures, and no API key in logs."""

import logging
import math
import re
import time

import requests
from requests.adapters import HTTPAdapter

from movierec.config import TMDB_API_BASE_URL, TMDB_API_KEY, TMDB_IMAGE_BASE_URL

logger = logging.getLogger(__name__)
# urllib3 logs request URLs, query string and api_key included, at DEBUG level.
logging.getLogger("urllib3").setLevel(logging.WARNING)

# (connect, read) seconds. A page makes several calls, so a stalled TMDb must fail fast.
TIMEOUT = (3.05, 5)
# After a failure, further calls fail immediately for a while instead of each page
# waiting for its own timeouts.
COOLDOWN_SECONDS = 30.0
TRAILER_TYPES = ("Trailer", "Teaser")
# With language=tr-TR, TMDb only returns Turkish videos unless other languages are asked
# for, and most films have no Turkish trailer; "null" adds videos without a language.
VIDEO_LANGUAGES = "tr,en,null"
# Connections kept open to TMDb. The poster workers (8) and gunicorn's request threads (4)
# share the session; urllib3's default of 10 would open and drop the extra ones each time.
POOL_SIZE = 16
# A YouTube video id; the trailer key goes into i.ytimg.com and youtube-nocookie.com URLs.
_YOUTUBE_KEY = re.compile(r"[A-Za-z0-9_-]{1,64}")
_YEAR = re.compile(r"[0-9]{4}")

_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_maxsize=POOL_SIZE))
_cooldown_until = 0.0


class TmdbUnavailableError(Exception):
    """TMDb is not configured, unreachable, or rejected the request."""


def is_configured() -> bool:
    """Return True when a TMDb API key is set."""
    return bool(TMDB_API_KEY)


def get(path: str, **params) -> dict | None:
    """GET a TMDb v3 path; None on 404, TmdbUnavailableError on any other failure."""
    global _cooldown_until
    if not is_configured():
        raise TmdbUnavailableError("TMDb API key is not configured")
    if time.monotonic() < _cooldown_until:
        raise TmdbUnavailableError(path)

    try:
        response = _session.get(
            TMDB_API_BASE_URL + path,
            params={"api_key": TMDB_API_KEY, **params},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        logger.warning("TMDb %s failed: %s", path, type(exc).__name__)
        _cooldown_until = time.monotonic() + COOLDOWN_SECONDS
        # "from None" keeps the original exception, whose message holds the URL with the
        # key, out of chained tracebacks.
        raise TmdbUnavailableError(path) from None

    if response.status_code == 404:
        return None
    if not response.ok:
        logger.warning("TMDb %s returned HTTP %s", path, response.status_code)
        _cooldown_until = time.monotonic() + COOLDOWN_SECONDS
        raise TmdbUnavailableError(path)
    try:
        data = response.json()
    except ValueError:
        data = None
    # Every TMDb answer is a JSON object; anything else is as broken as invalid JSON.
    if not isinstance(data, dict):
        logger.warning("TMDb %s returned invalid JSON", path)
        raise TmdbUnavailableError(path)
    return data


# Readers for the fields of an answer. get() guarantees a JSON object, not the shape of its
# fields; with these an odd field counts as missing instead of failing the page.


def record(value) -> dict:
    """A TMDb object field, or {} when it is missing or not an object."""
    return value if isinstance(value, dict) else {}


def records(value) -> list[dict]:
    """The objects of a TMDb list field; a field of another shape counts as empty."""
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def text(value) -> str:
    """A TMDb text field, or "" when it is missing or not text."""
    return value if isinstance(value, str) else ""


def number(value) -> float:
    """A TMDb numeric field as a finite float; a missing or odd value counts as 0."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def valid_id(value) -> int | None:
    """A TMDb id (a positive integer), or None; links and caches need a real id."""
    return value if type(value) is int and value > 0 else None


def image_url(path, size: str) -> str | None:
    """Build a TMDb image URL such as .../t/p/w342/abc.jpg, or None without a path."""
    return f"{TMDB_IMAGE_BASE_URL}{size}{path}" if text(path).startswith("/") else None


def release_year(data: dict) -> int:
    """Year of a TMDb movie or credit from its 'YYYY-MM-DD' release_date, 0 when unknown."""
    release_date = text(data.get("release_date"))
    return int(release_date[:4]) if _YEAR.match(release_date) else 0


def pick_trailer(videos) -> str | None:
    """Return the YouTube key of the best video: official trailer, any trailer, then teasers.

    Within each of those tiers a Turkish video wins; otherwise TMDb's order is kept.
    """
    youtube = [
        video
        for video in records(videos)
        if video.get("site") == "YouTube" and _YOUTUBE_KEY.fullmatch(text(video.get("key")))
    ]
    for video_type in TRAILER_TYPES:
        of_type = [video for video in youtube if video.get("type") == video_type]
        for tier in ([video for video in of_type if video.get("official")], of_type):
            if tier:
                turkish = [video for video in tier if video.get("iso_639_1") == "tr"]
                return (turkish or tier)[0]["key"]
    return None
