import re

import pytest

from movierec import config


@pytest.mark.parametrize("raw", ["", "   ", "your_secret_key_here", "dev_secret_key_!@#$"])
def test_placeholder_secret_key_is_replaced_by_random_key(raw):
    key, ephemeral = config._resolve_secret_key(raw)

    assert ephemeral is True
    assert re.fullmatch(r"[0-9a-f]{64}", key)


def test_random_secret_key_differs_per_call():
    assert config._resolve_secret_key("")[0] != config._resolve_secret_key("")[0]


def test_real_secret_key_is_kept():
    assert config._resolve_secret_key(" 3f9c1e-real-key ") == ("3f9c1e-real-key", False)


@pytest.mark.parametrize("raw", [None, "", "  ", "your_api_key", " your_tmdb_api_key"])
def test_placeholder_tmdb_key_counts_as_missing(raw):
    assert config._resolve_tmdb_key(raw) is None


def test_real_tmdb_key_is_stripped():
    assert config._resolve_tmdb_key(" abc123 ") == "abc123"


def test_security_headers_are_set(client):
    headers = client.get("/").headers

    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_session_cookie_is_httponly_and_samesite_lax(client):
    # The session only carries flash messages; an empty search flashes one.
    response = client.post("/search_movie_redirect", data={"search_query": ""})

    cookie = response.headers["Set-Cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=Lax" in cookie
