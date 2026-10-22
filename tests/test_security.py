import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from pages import Page, body_text

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
    assert headers["Cross-Origin-Opener-Policy"] == "same-origin"
    assert headers["Cross-Origin-Resource-Policy"] == "same-origin"


def test_session_cookie_is_httponly_and_samesite_lax(client):
    # The session only carries flash messages; an empty search flashes one.
    response = client.post("/search_movie_redirect", data={"search_query": ""})

    cookie = response.headers["Set-Cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=Lax" in cookie


def test_links_that_open_a_new_tab_cannot_reach_back_to_the_opener():
    templates = Path(__file__).resolve().parents[1] / "templates"
    for template in templates.glob("*.html"):
        for tag in re.findall(r"<a\b[^>]*>", template.read_text(encoding="utf-8")):
            if 'target="_blank"' in tag:
                assert re.search(r'rel="[^"]*\bnoopener\b', tag), (template.name, tag)


# The policy every response carries. The pages have no inline scripts, styles or handlers
# (tests/test_layout.py checks the templates), so none of the directives needs 'unsafe-inline'.
CSP = {
    "default-src": ["'self'"],
    "script-src": ["'self'"],
    "style-src": ["'self'", "https://fonts.googleapis.com"],
    "font-src": ["https://fonts.gstatic.com"],
    "img-src": [
        "'self'",
        "https://image.tmdb.org",
        "https://i.ytimg.com",
        "https://www.themoviedb.org",
        "data:",
    ],
    "frame-src": ["https://www.youtube-nocookie.com"],
    "connect-src": ["'self'"],
    "base-uri": ["'none'"],
    "form-action": ["'self'"],
    "frame-ancestors": ["'none'"],
    "object-src": ["'none'"],
}
STATIC = Path(__file__).resolve().parents[1] / "static"


def _policy(header):
    directives = (part.split() for part in header.split(";") if part.strip())
    return {name: sources for name, *sources in directives}


def _allows(directive, url):
    """True when the CSP directive lets the page load `url` (a relative URL is same-origin)."""
    sources = CSP[directive]
    if url.startswith("data:"):
        return "data:" in sources
    parts = urlsplit(url)
    if not parts.scheme and not parts.netloc:
        return "'self'" in sources
    return f"{parts.scheme}://{parts.netloc}" in sources


@pytest.mark.parametrize(
    "path", ["/", "/?tur=Action", "/movie/155", "/my_lists", "/nope", "/api/titles", "/health"]
)
def test_every_response_carries_the_content_security_policy(client, path):
    response = client.get(path)

    assert _policy(response.headers["Content-Security-Policy"]) == CSP


def test_error_pages_and_redirects_carry_the_policy_too(client):
    for response in (
        client.put("/"),
        client.post("/search_movie_redirect", data={"search_query": ""}),
    ):
        assert _policy(response.headers["Content-Security-Policy"]) == CSP


@pytest.fixture
def tmdb_pages(fake, cold_details, person_client, monkeypatch):
    """Every page type with TMDb answering: posters, a trailer and a profile photo."""
    from movierec import tmdb

    answer = {
        "poster_path": "/p.jpg",
        "videos": {"results": [{"site": "YouTube", "type": "Trailer", "key": "abc123"}]},
    }
    monkeypatch.setattr(tmdb, "get", lambda path, **params: answer)
    return {
        path: body_text(person_client.get(path))
        for path in ("/", "/?film=Avatar", "/movie/155", "/actor/Christian+Bale", "/my_lists")
    }


def test_everything_the_pages_load_is_allowed_by_the_policy(tmdb_pages):
    loads = 0
    for path, body in tmdb_pages.items():
        page = Page(body)
        # The card <template> of the lists page holds an <img> that my_lists.js gives a src.
        resources = [("img-src", tag["src"]) for tag in page.all("img") if "src" in tag]
        resources += [("script-src", tag["src"]) for tag in page.all("script")]
        resources += [("style-src", tag["href"]) for tag in page.all("link", rel="stylesheet")]
        resources += [("img-src", tag["href"]) for tag in page.all("link", rel="icon")]
        resources += [("form-action", tag["action"]) for tag in page.all("form")]
        for directive, url in resources:
            assert _allows(directive, url), (path, directive, url)
        loads += len(resources)
    assert {
        urlsplit(src).netloc for src in re.findall(r'src="([^"]+)"', "".join(tmdb_pages.values()))
    } >= {
        "image.tmdb.org",
        "i.ytimg.com",
        "www.themoviedb.org",
    }
    assert loads


def test_rendered_pages_hold_nothing_inline_that_the_policy_would_block(tmdb_pages):
    # Without 'unsafe-inline' the browser drops inline scripts, styles and event handlers.
    for path, body in tmdb_pages.items():
        page = Page(body)
        for tag, attrs in page.tags:
            assert not [name for name in attrs if name.startswith("on")], (path, tag, attrs)
            assert "style" not in attrs, (path, tag, attrs)
            for name in ("href", "src", "action"):
                assert not (attrs.get(name) or "").lower().startswith("javascript:"), (path, tag)
            if tag == "script":
                assert attrs.get("src"), path
        assert not page.find("style"), path


def test_scripts_and_stylesheets_reach_only_origins_the_policy_allows():
    # movie_detail.js builds the trailer's iframe; nothing else is fetched from another site.
    scripts = list((STATIC / "js").glob("*.js"))
    assert scripts
    origins = {
        origin
        for script in scripts
        for origin in re.findall(r"https://[\w.-]+", script.read_text(encoding="utf-8"))
    }
    assert origins == {"https://www.youtube-nocookie.com"}
    assert all(_allows("frame-src", origin) for origin in origins)
    # The stylesheets draw their small images as data: URLs and import nothing.
    for sheet in (STATIC / "css").glob("*.css"):
        css = sheet.read_text(encoding="utf-8")
        assert "@import" not in css, sheet.name
        for url in re.findall(r"url\(\s*[\"']?([^\"')]+)", css):
            assert _allows("img-src", url), (sheet.name, url[:40])


def test_powerful_browser_features_are_switched_off(client):
    policy = client.get("/").headers["Permissions-Policy"]

    features = dict(part.strip().split("=") for part in policy.split(","))
    assert features == {
        "camera": "()",
        "microphone": "()",
        "geolocation": "()",
        "payment": "()",
        "usb": "()",
    }


def test_hsts_is_sent_only_when_the_site_is_served_over_https(client, app_module, monkeypatch):
    assert "Strict-Transport-Security" not in client.get("/").headers

    monkeypatch.setitem(app_module.app.config, "SESSION_COOKIE_SECURE", True)

    assert client.get("/").headers["Strict-Transport-Security"] == "max-age=31536000"


def test_an_oversized_request_body_is_refused_before_it_is_read(client):
    # Werkzeug reads a urlencoded form whole; without a limit one request could fill memory.
    response = client.post("/search_movie_redirect", data={"search_query": "q" * 20_000})

    assert response.status_code == 413
    assert "Gönderilen istek çok büyük." in body_text(response)


def test_a_normal_header_search_fits_the_body_limit(client):
    response = client.post("/search_movie_redirect", data={"search_query": "q" * 5000})

    assert response.status_code == 302
