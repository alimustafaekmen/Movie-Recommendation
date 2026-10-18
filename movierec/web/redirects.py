"""Redirect helpers for the movies blueprint; API_PREFIXES also picks JSON error pages."""

from urllib.parse import urlsplit

from flask import redirect, request, url_for

API_PREFIXES = ("/api/",)
# Never send a user back to these: they are POST-only routes or JSON endpoints.
_POST_ONLY_PREFIXES = (*API_PREFIXES, "/search_movie_redirect")


def safe_redirect_target(target):
    """Return target if it is a local path that can be opened with GET, else None."""
    if not target or not target.startswith("/") or target.startswith("//"):
        return None
    # Browsers read a backslash as a slash and drop tabs and newlines, so "/\evil.example"
    # or "/<TAB>/evil.example" would lead to another site.
    if "\\" in target or any(char < " " or char == "\x7f" for char in target):
        return None
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or parts.path.startswith(_POST_ONLY_PREFIXES):
        return None
    return target


def redirect_back():
    """Redirect to the referring page when it belongs to this site, otherwise home."""
    target = None
    if request.referrer:
        parts = urlsplit(request.referrer)
        if parts.netloc == request.host:
            target = safe_redirect_target(parts.path + (f"?{parts.query}" if parts.query else ""))
    return redirect(target or url_for("movies.home"))
