"""The language of a request: ?lang=, then the stored choice, then the browser's.

Only a choice is stored: the header switch and shared links carry ?lang=, which goes into a
cookie. Without one the browser's Accept-Language decides again on every visit.
"""

from urllib.parse import quote, urlencode

from flask import current_app, g, request, url_for

from movierec.i18n import DEFAULT_LANGUAGE, LANGUAGES, translate, translate_count

COOKIE_NAME = "lang"
COOKIE_MAX_AGE_SECONDS = 365 * 24 * 60 * 60


def get_lang() -> str:
    """Return the language of this request, worked out once and kept in g."""
    if "lang" not in g:
        g.lang = _pick_language()
    return g.lang


def _pick_language() -> str:
    """?lang=, then the cookie, then the browser's Accept-Language; English otherwise."""
    for value in (request.args.get("lang"), request.cookies.get(COOKIE_NAME)):
        if value in LANGUAGES:
            return value
    # The first entry whose language is Turkish or English, in falling q order. best_match
    # would prefer a later exact "en" over a regional "tr-TR". A wildcard is no choice.
    for value, quality in request.accept_languages:
        language = value.replace("_", "-").split("-")[0].casefold()
        if quality > 0 and language in LANGUAGES:
            return language
    return DEFAULT_LANGUAGE


def remember_language(response):
    """Store a ?lang= choice and tell caches when the answer depends on the language."""
    chosen = request.args.get("lang")
    if chosen in LANGUAGES and request.cookies.get(COOKIE_NAME) != chosen:
        response.set_cookie(
            COOKIE_NAME,
            chosen,
            max_age=COOKIE_MAX_AGE_SECONDS,
            samesite="Lax",
            secure=current_app.config["SESSION_COOKIE_SECURE"],
            httponly=True,
        )
    # get_lang() keeps the language in g, so this marks exactly the answers that read it; the
    # suggestion lists and static files stay one copy for every language.
    if "lang" in g:
        response.vary.update(("Accept-Language", "Cookie"))
    return response


def lang_url(code: str) -> str:
    """Return the URL of this page in another language, for the header switch."""
    # After the header search's POST there is no page to ask for again, so the switch on its
    # error page leads home.
    if request.method != "GET":
        return url_for("movies.home", lang=code)
    query = [(key, value) for key, value in request.args.items(multi=True) if key != "lang"]
    query.append(("lang", code))
    # request.path is decoded ("/actor/Zoë Saldaña"); quote() encodes it again.
    return f"{quote(request.script_root + request.path)}?{urlencode(query)}"


def t(key: str, **params) -> str:
    """Return a catalogue text in the language of this request."""
    return translate(get_lang(), key, **params)


def t_count(key: str, count: int, **params) -> str:
    """Return the singular or plural catalogue text for count in the language of this request."""
    return translate_count(get_lang(), key, count, **params)


def t_message(item: dict) -> str:
    """Return a movierec.i18n.message(), e.g. an engine warning, as text in this language."""
    return t(item["key"], **item["params"])
