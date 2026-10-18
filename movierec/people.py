"""Director and actor pages: TMDb person search, profile and filmography."""

import logging

from movierec import tmdb

log = logging.getLogger(__name__)

MAX_NAME_LENGTH = 100
MOVIE_LIMIT = 24
OVERVIEW_PREVIEW_CHARS = 120
DEPARTMENTS = {"director": "Directing", "actor": "Acting"}


def find_person(name: str, person_type: str) -> dict | None:
    """Return the person page data, or None if TMDb knows nobody by that name.

    Raises tmdb.TmdbUnavailable when TMDb is not configured or cannot be reached.
    """
    name = name.strip()
    # Real names are short; longer input is junk and would only produce a 414 upstream.
    if not name or len(name) > MAX_NAME_LENGTH:
        return None

    person_id = _search_person(name, person_type)
    if person_id is None:
        return None
    details = _fetch_person(person_id)
    if details is None:
        return None

    person = _profile(details)
    person["movies"] = _format_credits(details.get("movie_credits") or {}, person_type)
    person["person_type_tr"] = "Yönetmen" if person_type == "director" else "Oyuncu"
    person["page_title_name"] = name
    return person


def short_overview(overview: str | None) -> str:
    """Shorten a filmography overview to OVERVIEW_PREVIEW_CHARS, adding '…' only when cut."""
    if not overview:
        return "Özet bulunmuyor."
    if len(overview) <= OVERVIEW_PREVIEW_CHARS:
        return overview
    return overview[: OVERVIEW_PREVIEW_CHARS - 1].rstrip() + "…"


def _search_person(name: str, person_type: str) -> int | None:
    data = tmdb.get("/search/person", query=name, language="tr-TR", include_adult="false")
    results = (data or {}).get("results") or []
    if not results:
        log.info("No TMDb person matches %r", name)
        return None

    # A common name returns several people; prefer the one known for the requested
    # role and fall back to TMDb's top hit.
    best = results[0]
    department = DEPARTMENTS.get(person_type)
    if department is not None:
        best = next((r for r in results if r.get("known_for_department") == department), best)
    return best.get("id")


def _fetch_person(person_id: int) -> dict | None:
    return tmdb.get(f"/person/{person_id}", language="tr-TR", append_to_response="movie_credits")


def _profile(details: dict) -> dict:
    profile_path = details.get("profile_path")
    homepage = details.get("homepage") or ""
    return {
        "id": details.get("id"),
        "name": details.get("name"),
        "biography": details.get("biography") or "Biyografi bilgisi bulunmamaktadır.",
        "birthday": details.get("birthday"),
        "deathday": details.get("deathday"),
        "place_of_birth": details.get("place_of_birth"),
        "profile_path": profile_path,
        "known_for_department": details.get("known_for_department"),
        "profile_image_url": tmdb.image_url(profile_path, "h632"),
        # The page links to it, so only web addresses pass; a javascript: URL would run on click.
        "homepage": homepage if homepage.startswith(("http://", "https://")) else None,
    }


def _format_credits(movie_credits: dict, person_type: str) -> list[dict]:
    is_director = person_type == "director"
    credits = movie_credits.get("crew" if is_director else "cast") or []
    # Credits without a poster render as empty cards, so they are left out.
    credits = [
        credit
        for credit in credits
        if credit.get("poster_path") and (not is_director or credit.get("job") == "Director")
    ]
    credits.sort(key=_newest_first_key, reverse=True)

    movies, seen_ids = [], set()
    for credit in credits:
        if credit.get("id") in seen_ids:
            continue
        seen_ids.add(credit.get("id"))
        movies.append(_credit_card(credit, person_type))
        if len(movies) >= MOVIE_LIMIT:
            break
    return movies


def _newest_first_key(credit: dict) -> tuple:
    # TMDb dates are ISO 'YYYY-MM-DD', so string order is date order; undated films sort last.
    return credit.get("release_date") or "", credit.get("popularity") or 0


def _credit_card(credit: dict, person_type: str) -> dict:
    card = {
        "id": credit.get("id"),
        "title": credit.get("title") or credit.get("name"),
        "poster_url": tmdb.image_url(credit.get("poster_path"), "w342"),
        "release_date": credit.get("release_date"),
        "overview": short_overview(credit.get("overview")),
        "vote_average": credit.get("vote_average", 0),
    }
    if person_type == "director":
        card["job"] = "Yönetmen"
    elif person_type == "actor":
        card["character"] = credit.get("character", "Bilinmiyor")
    return card
