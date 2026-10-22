"""Redirect helpers for the movies blueprint; API_PREFIXES also picks JSON error pages."""

from urllib.parse import urlsplit

from flask import redirect, request, url_for
from werkzeug.wrappers import Response

API_PREFIXES = ("/api/",)
# Never send a user back to these: they are POST-only routes or JSON endpoints.
_NO_REDIRECT_PREFIXES = (*API_PREFIXES, "/search_movie_redirect")


def safe_redirect_target(target: str | None) -> str | None:
    """Return target if it is a local path that can be opened with GET, else None."""
    if not target:
        return None
    # A local path starts with one slash; "//evil.example" is another site.
    if not target.startswith("/") or target.startswith("//"):
        return None
    # Browsers read a backslash as a slash and drop tabs and newlines, so "/\evil.example"
    # or "/<TAB>/evil.example" would lead to another site.
    if "\\" in target:
        return None
    for char in target:
        if char < " " or char == "\x7f":  # control characters and DEL
            return None
    parts = urlsplit(target)
    if parts.scheme or parts.netloc:
        return None
    # POST-only routes and JSON endpoints cannot be opened as a page.
    if parts.path.startswith(_NO_REDIRECT_PREFIXES):
        return None
    return target


def redirect_back() -> Response:
    """Redirect to the referring page when it belongs to this site, otherwise home."""
    home_url = url_for("movies.home")
    try:
        referrer = urlsplit(request.referrer or "")
    except ValueError:
        # A broken header, e.g. "http://[": nothing to go back to.
        return redirect(home_url)
    if referrer.netloc != request.host:
        return redirect(home_url)
    path = referrer.path
    if referrer.query:
        path = f"{path}?{referrer.query}"
    target = safe_redirect_target(path)
    if target is None:
        return redirect(home_url)
    return redirect(target)
