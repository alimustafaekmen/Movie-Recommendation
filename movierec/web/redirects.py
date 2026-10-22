"""Redirect helpers for the movies blueprint; API_PREFIXES also picks JSON error pages."""

from urllib.parse import urlsplit

from flask import redirect, request, url_for
from werkzeug.wrappers import Response

API_PREFIXES = ("/api/",)
# Never send a user back to these: they are POST-only routes or JSON endpoints.
_NO_REDIRECT_PREFIXES = (*API_PREFIXES, "/search_movie_redirect")


def safe_redirect_target(target: str | None) -> str | None:
    """Return target if it is a local path that can be opened with GET, else None."""
    if not target or not target.startswith("/") or target.startswith("//"):
        return None
    # Browsers read a backslash as a slash and drop tabs and newlines, so "/\evil.example"
    # or "/<TAB>/evil.example" would lead to another site.
    if "\\" in target or any(char < " " or char == "\x7f" for char in target):
        return None
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or parts.path.startswith(_NO_REDIRECT_PREFIXES):
        return None
    return target


def redirect_back() -> Response:
    """Redirect to the referring page when it belongs to this site, otherwise home."""
    target = None
    try:
        parts = urlsplit(request.referrer or "")
    except ValueError:
        # A broken header, e.g. "http://[": nothing to go back to.
        parts = None
    if parts is not None and parts.netloc == request.host:
        target = safe_redirect_target(parts.path + (f"?{parts.query}" if parts.query else ""))
    return redirect(target or url_for("movies.home"))
