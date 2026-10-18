"""Minimal TMDb v3 client: short timeouts, a cooldown after failures, and no API key in logs."""

import logging
import time

import requests

from movierec.config import TMDB_API_BASE_URL, TMDB_API_KEY, TMDB_POSTER_BASE_URL

log = logging.getLogger(__name__)
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

_session = requests.Session()
_cooldown_until = 0.0


class TmdbUnavailable(Exception):
    """TMDb is not configured, unreachable, or rejected the request."""


def is_configured() -> bool:
    """Return True when a TMDb API key is set."""
    return bool(TMDB_API_KEY)


def get(path: str, **params) -> dict | None:
    """GET a TMDb v3 path; None on 404, TmdbUnavailable on any other failure."""
    global _cooldown_until
    if not is_configured():
        raise TmdbUnavailable("TMDb API key is not configured")
    if time.monotonic() < _cooldown_until:
        raise TmdbUnavailable(path)

    try:
        response = _session.get(
            TMDB_API_BASE_URL + path,
            params={"api_key": TMDB_API_KEY, **params},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        log.warning("TMDb %s failed: %s", path, type(exc).__name__)
        _cooldown_until = time.monotonic() + COOLDOWN_SECONDS
        # "from None" keeps the original exception, whose message holds the URL with the
        # key, out of chained tracebacks.
        raise TmdbUnavailable(path) from None

    if response.status_code == 404:
        return None
    if not response.ok:
        log.warning("TMDb %s returned HTTP %s", path, response.status_code)
        _cooldown_until = time.monotonic() + COOLDOWN_SECONDS
        raise TmdbUnavailable(path)
    try:
        return response.json()
    except ValueError:
        log.warning("TMDb %s returned invalid JSON", path)
        raise TmdbUnavailable(path) from None


def image_url(path: str | None, size: str) -> str | None:
    """Build a TMDb image URL such as .../t/p/w342/abc.jpg, or None without a path."""
    return f"{TMDB_POSTER_BASE_URL}{size}{path}" if path else None


def pick_trailer(videos: list[dict]) -> str | None:
    """Return the YouTube key of the best video: official trailer, any trailer, then teasers.

    Within each of those tiers a Turkish video wins; otherwise TMDb's order is kept.
    """
    youtube = [video for video in videos if video.get("site") == "YouTube"]
    for video_type in TRAILER_TYPES:
        of_type = [video for video in youtube if video.get("type") == video_type]
        for tier in ([video for video in of_type if video.get("official")], of_type):
            if tier:
                turkish = [video for video in tier if video.get("iso_639_1") == "tr"]
                return (turkish or tier)[0].get("key")
    return None
