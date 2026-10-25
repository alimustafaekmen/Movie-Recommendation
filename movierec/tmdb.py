"""Minimal TMDb v3 client: short timeouts, a cooldown after failures, and no API key in logs."""

import logging
import math
import re
import time

import requests
from requests.adapters import HTTPAdapter

from movierec.config import TMDB_API_BASE_URL, TMDB_API_KEY, TMDB_IMAGE_BASE_URL

logger = logging.getLogger(__name__)
# urllib3 logs request URLs, query string and api_key included, at DEBUG level and in some
# warnings, such as one about a malformed header.
logging.getLogger("urllib3").setLevel(logging.ERROR)

# A page makes several calls, so a stalled TMDb must fail fast. 3.05 is just over TCP's 3 s
# retransmission window, as the requests docs advise for connect timeouts.
CONNECT_TIMEOUT_SECONDS = 3.05
READ_TIMEOUT_SECONDS = 5
# After a failure, further calls fail immediately for a while instead of each page
# waiting for its own timeouts.
COOLDOWN_SECONDS = 30.0
TRAILER_TYPES = ("Trailer", "Teaser")
# The videos asked for with each page language (include_video_language). With a language TMDb
# only returns videos in it unless others are asked for, and most films have no Turkish
# trailer; "null" adds videos without a language.
VIDEO_LANGUAGES = {"tr": "tr,en,null", "en": "en,null"}
# Connections kept open to TMDb. The poster workers (8) and gunicorn's request threads (4)
# share the session; urllib3's default of 10 would open and drop the extra ones each time.
POOL_SIZE = 16
# A YouTube video id: letters, digits, "_" and "-".
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
    """GET a TMDb v3 path; None on 404, TmdbUnavailableError on any other failure.

    params: TMDb query parameters, e.g. language="tr-TR"; the API key is added here.
    """
    global _cooldown_until
    if not is_configured():
        raise TmdbUnavailableError("TMDb API key is not configured")
    if time.monotonic() < _cooldown_until:
        raise TmdbUnavailableError(path)

    query_params = {"api_key": TMDB_API_KEY}
    query_params.update(params)
    try:
        response = _session.get(
            TMDB_API_BASE_URL + path,
            params=query_params,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
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
    """Return a TMDb object field, or {} when it is missing or not an object."""
    return value if isinstance(value, dict) else {}


def records(value) -> list[dict]:
    """Return the objects of a TMDb list field; a field of another shape counts as empty."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def text(value) -> str:
    """Return a TMDb text field, or "" when it is missing or not text."""
    return value if isinstance(value, str) else ""


def number(value) -> float:
    """Return a TMDb numeric field as a finite float; a missing or odd value counts as 0."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    # float() also accepts "nan" and "inf"; those count as missing too.
    if not math.isfinite(result):
        return 0.0
    return result


def valid_id(value) -> int | None:
    """Return a TMDb id (a positive integer), or None; links and caches need a real id."""
    # type(), not isinstance(): JSON true/false are bools, which isinstance counts as int.
    if type(value) is not int or value <= 0:
        return None
    return value


def image_url(path, size: str) -> str | None:
    """Return a TMDb image URL such as .../t/p/w342/abc.jpg, or None without a path.

    path: a poster_path or profile_path field, e.g. "/abc.jpg".
    """
    if not text(path).startswith("/"):
        return None
    return f"{TMDB_IMAGE_BASE_URL}{size}{path}"


def title(data: dict) -> str:
    """Return a TMDb film's title as the pages show it: its original title, else its title.

    A film has one title in both languages, so it reads the same whatever the page language.
    """
    return text(data.get("original_title")) or text(data.get("title"))


def release_year(data: dict) -> int:
    """Return the year of a TMDb movie or credit from its 'YYYY-MM-DD' release_date, or 0."""
    release_date = text(data.get("release_date"))
    # "2008-07-16" -> 2008; an empty or odd date counts as unknown.
    if not _YEAR.match(release_date):
        return 0
    return int(release_date[:4])


def pick_trailer(videos, lang: str) -> str | None:
    """Return the YouTube key of the best video, or None.

    videos: the "results" list of TMDb's videos answer. Trailers beat teasers and an official
    video beats the others of its type; among those one in the page language (lang) wins,
    otherwise TMDb's order is kept.
    """
    youtube_videos = []
    for video in records(videos):
        # The key goes into i.ytimg.com and youtube-nocookie.com URLs, so it must be a plain id.
        if video.get("site") == "YouTube" and _YOUTUBE_KEY.fullmatch(text(video.get("key"))):
            youtube_videos.append(video)
    for video_type in TRAILER_TYPES:
        of_type = [video for video in youtube_videos if video.get("type") == video_type]
        if not of_type:
            continue
        official = [video for video in of_type if video.get("official")]
        candidates = official if official else of_type
        for video in candidates:
            if video.get("iso_639_1") == lang:
                return video["key"]
        return candidates[0]["key"]
    return None
