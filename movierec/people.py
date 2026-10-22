"""Director and actor pages: TMDb person search, profile and filmography."""

import logging
import re

from movierec import tmdb

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 100
MOVIE_LIMIT = 24
# A person page costs two TMDb round trips (about 400 ms), so found people are remembered.
# Emptied once full, like the poster cache: simpler than LRU and still bounded.
PERSON_CACHE_LIMIT = 256
# Cast credits where the person appears as themselves (talk shows, making-ofs, documentaries)
# or only in old footage; they are not roles and crowd out the actual films.
_NOT_A_ROLE = re.compile(r"^\s*(self|himself|herself|themselves)\b|archiv(e|al) footage", re.I)
PERSON_TYPE_DEPARTMENTS = {"director": "Directing", "actor": "Acting"}
# Turkish labels for TMDb's known_for_department; a department missing here gets no label.
DEPARTMENT_LABELS = {
    "Directing": "Yönetmen",
    "Acting": "Oyuncu",
    "Writing": "Senarist",
    "Production": "Yapımcı",
    "Camera": "Görüntü Yönetmeni",
    "Editing": "Kurgu",
    "Sound": "Ses",
    "Art": "Sanat Yönetmeni",
    "Costume & Make-Up": "Kostüm ve Makyaj",
    "Visual Effects": "Görsel Efekt",
    "Crew": "Ekip",
    "Lighting": "Işık",
}
# Page data per (name, person type), filled by find_person.
_person_cache: dict[tuple[str, str], dict] = {}


def find_person(name: str, person_type: str) -> dict | None:
    """Return the person page data, or None if TMDb knows nobody by that name.

    Raises tmdb.TmdbUnavailableError when TMDb is not configured or cannot be reached.
    """
    name = name.strip()
    # Real names are short; longer input is junk and would only produce a 414 upstream.
    if not name or len(name) > MAX_NAME_LENGTH:
        return None

    key = (name, person_type)
    person = _person_cache.get(key)
    if person is None:
        person = _lookup_person(name, person_type)
        # Misses and failures are not remembered, so a person TMDb adds later is found.
        if person is None:
            return None
        if len(_person_cache) >= PERSON_CACHE_LIMIT:
            _person_cache.clear()
        _person_cache[key] = person
    return person


def _lookup_person(name: str, person_type: str) -> dict | None:
    person_id = _search_person(name, person_type)
    if person_id is None:
        return None
    details = _fetch_person(person_id)
    if details is None:
        return None

    person = _profile(details)
    person["movies"] = _format_credits(tmdb.record(details.get("movie_credits")), person_type)
    person["person_type_tr"] = "Yönetmen" if person_type == "director" else "Oyuncu"
    return person


def _search_person(name: str, person_type: str) -> int | None:
    data = tmdb.get("/search/person", query=name, language="tr-TR", include_adult="false")
    results = [
        result
        for result in tmdb.records((data or {}).get("results"))
        if tmdb.valid_id(result.get("id"))
    ]
    if not results:
        logger.info("No TMDb person matches %r", name)
        return None

    # A common name returns several people; prefer the one known for the requested
    # role and fall back to TMDb's top hit.
    best = results[0]
    department = PERSON_TYPE_DEPARTMENTS.get(person_type)
    if department is not None:
        best = next((r for r in results if r.get("known_for_department") == department), best)
    return best["id"]


def _fetch_person(person_id: int) -> dict | None:
    return tmdb.get(f"/person/{person_id}", language="tr-TR", append_to_response="movie_credits")


def _profile(details: dict) -> dict:
    homepage = tmdb.text(details.get("homepage"))
    profile = {
        "id": tmdb.valid_id(details.get("id")),
        "name": tmdb.text(details.get("name")),
        "biography": tmdb.text(details.get("biography")) or "Biyografi bilgisi bulunmamaktadır.",
        "birthday": tmdb.text(details.get("birthday")),
        "deathday": tmdb.text(details.get("deathday")),
        "place_of_birth": tmdb.text(details.get("place_of_birth")),
        "known_for_department": tmdb.text(details.get("known_for_department")),
        "profile_image_url": tmdb.image_url(details.get("profile_path"), "h632"),
        # The page links to it, so only web addresses pass; a javascript: URL would run on click.
        "homepage": homepage if homepage.startswith(("http://", "https://")) else None,
    }
    label = DEPARTMENT_LABELS.get(profile["known_for_department"])
    if label:
        profile["known_for_label"] = label
    return profile


def _format_credits(movie_credits: dict, person_type: str) -> list[dict]:
    is_director = person_type == "director"
    # Credits without a poster render as empty cards, and a card needs an id to link to.
    credits = [
        credit
        for credit in tmdb.records(movie_credits.get("crew" if is_director else "cast"))
        if tmdb.valid_id(credit.get("id"))
        and tmdb.image_url(credit.get("poster_path"), "w342")
        and (credit.get("job") == "Director" if is_director else _is_a_role(credit))
    ]
    credits.sort(key=_best_known_first_key, reverse=True)

    movies, seen_ids = [], set()
    for credit in credits:
        if credit["id"] in seen_ids:
            continue
        seen_ids.add(credit["id"])
        movies.append(_credit_card(credit, person_type))
        if len(movies) >= MOVIE_LIMIT:
            break
    return movies


def _is_a_role(credit: dict) -> bool:
    return not _NOT_A_ROLE.search(tmdb.text(credit.get("character")))


def _best_known_first_key(credit: dict) -> tuple:
    # Vote counts grow with a film's audience over the years, so the well-known films lead;
    # TMDb's popularity is a trend score that would put unreleased films first. It only
    # breaks ties, e.g. between upcoming films that have no votes yet.
    return tmdb.number(credit.get("vote_count")), tmdb.number(credit.get("popularity"))


def _credit_card(credit: dict, person_type: str) -> dict:
    card = {
        "id": credit["id"],
        "title": tmdb.text(credit.get("title")) or tmdb.text(credit.get("name")),
        "poster_url": tmdb.image_url(credit.get("poster_path"), "w342"),
        "year": tmdb.release_year(credit),
        "vote_average": tmdb.number(credit.get("vote_average")),
    }
    if person_type == "actor":
        card["character"] = tmdb.text(credit.get("character"))
    return card
