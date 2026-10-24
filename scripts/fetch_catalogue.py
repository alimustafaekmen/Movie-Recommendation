"""Fetch a fresh catalogue from TMDb in the format of the Kaggle TMDB 5000 CSVs.

The films are the ids of the Kaggle set plus, for each release year from FIRST_YEAR on,
the PER_YEAR most-voted films with at least MIN_VOTES_THRESHOLD votes, less the erotic, porn
and shock films (EXCLUDED_KEYWORDS). Every film is fetched on the same day, so their vote
counts and popularity can be compared. A full run makes about 15,000 requests and takes about
15 minutes. It needs TMDB_API_KEY in .env.

The files land in dataset/refreshed/, which git ignores; DATASET_DIR=dataset/refreshed
points the app at them. TMDb's terms allow keeping the data for at most 6 months, so run
this again before then.

    python scripts/fetch_catalogue.py
    python scripts/fetch_catalogue.py --limit 50 --out /tmp/trial
"""

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from movierec import tmdb  # noqa: E402
from movierec.config import MIN_VOTES_THRESHOLD, TMDB_API_BASE_URL, TMDB_API_KEY  # noqa: E402
from movierec.dataset import CAST_LIMIT, CREDIT_COLUMNS, MOVIE_COLUMNS  # noqa: E402

# The Kaggle set ends in late 2016 and holds only part of each earlier year, so it misses
# well-known films such as John Wick (2014). The years from 1990 on are filled in too.
FIRST_YEAR = 1990
PER_YEAR = 400
# The seed ids always come from the Kaggle file, even when DATASET_DIR points the app at
# the refreshed catalogue.
KAGGLE_MOVIES_CSV = REPO / "dataset" / "tmdb_5000_movies.csv"
OUT_DIR = REPO / "dataset" / "refreshed"
# The catalogue is a showcase, so it leaves out the films TMDb tags as erotic, porn or shock
# films; TMDb's adult flag misses them. Tags such as "rape" or "pornography" also mark
# well-known dramas, so only the tags of these genres count.
EXCLUDED_KEYWORDS = frozenset(
    {
        "adult filmmaking",
        "erotic drama",
        "erotic fantasy",
        "erotic movie",
        "porn industry",
        "porno",
        "sexploitation",
        "soft porn",
        "softcore",
        "torture porn",
    }
)
# The films no such tag marks, picked one by one: explicit or offensive titles, sex films and
# porn documentaries, and notorious shock films.
# fmt: off
EXCLUDED_IDS = frozenset({
    692, 979, 1378, 1631, 1836, 3134, 4241, 7090, 7295, 8675,
    9539, 9696, 10068, 10358, 10842, 11845, 12228, 12517, 12703, 12838,
    13429, 13461, 13492, 13702, 13797, 14125, 14758, 15356, 16363, 18167,
    18392, 20712, 23966, 24593, 24836, 24922, 26715, 27274, 27297, 28005,
    31022, 33380, 33431, 33560, 36253, 38410, 39857, 40185, 43947, 48266,
    48636, 52462, 53064, 53932, 63197, 65599, 67911, 69778, 73861, 102840,
    112160, 123229, 132030, 162396, 171424, 178862, 179538, 207768, 215743, 216015,
    292431, 319314, 325848, 337167, 341174, 357096, 499726, 598133, 664413, 728949,
    776797, 829557, 829560, 833326, 1252309, 1288059,
})
# fmt: on
# Films per /discover/movie page, set by TMDb.
PAGE_SIZE = 20
DETAILS_PARAMS = {"language": "en-US", "append_to_response": "credits,keywords,translations"}
WORKERS = 6
# Across all workers; well under TMDb's limit of about 50 a second.
REQUESTS_PER_SECOND = 20
TIMEOUT = (tmdb.CONNECT_TIMEOUT_SECONDS, tmdb.READ_TIMEOUT_SECONDS)
# urllib3 logs every retry at WARNING with the request URL, api_key included.
logging.getLogger("urllib3").setLevel(logging.ERROR)


class FetchError(Exception):
    """A TMDb request failed for a reason other than a missing film."""


def make_session() -> requests.Session:
    """Return a pooled session that retries rate limits, server errors and lost connections."""
    # Retry waits as long as a 429 or 503 answer's Retry-After header asks. After the last
    # try it hands back the answer, so the error can name its HTTP status.
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=WORKERS))
    return session


class Client:
    """GETs TMDb paths through one session, at most REQUESTS_PER_SECOND across all threads."""

    def __init__(self, session):
        self.session = session
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def _wait_for_slot(self):
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + 1 / REQUESTS_PER_SECOND
        time.sleep(slot - now)

    def get(self, path: str, params: dict) -> dict | None:
        """GET a TMDb path; None on 404, FetchError on any other failure.

        params: TMDb query parameters; the API key is added here and never shows in errors.
        """
        self._wait_for_slot()
        try:
            response = self.session.get(
                TMDB_API_BASE_URL + path,
                params={"api_key": TMDB_API_KEY, **params},
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            # The exception's message holds the URL with the key; "from None" also keeps it
            # out of the traceback.
            raise FetchError(f"{path}: {type(exc).__name__}") from None
        if response.status_code == 404:
            return None
        if not response.ok:
            raise FetchError(f"{path}: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            data = None
        if not isinstance(data, dict):
            raise FetchError(f"{path}: invalid JSON")
        return data


def discover_year(client: Client, year: int, per_year: int) -> list[int]:
    """Return the ids of the per_year most-voted films of a release year with enough votes."""
    params = {
        "sort_by": "vote_count.desc",
        "primary_release_date.gte": f"{year}-01-01",
        "primary_release_date.lte": f"{year}-12-31",
        "vote_count.gte": MIN_VOTES_THRESHOLD,
    }
    ids = []
    for page in range(1, math.ceil(per_year / PAGE_SIZE) + 1):
        answer = client.get("/discover/movie", {**params, "page": page})
        if answer is None:
            raise FetchError("/discover/movie: HTTP 404")
        for film in tmdb.records(answer.get("results")):
            movie_id = tmdb.valid_id(film.get("id"))
            if movie_id and tmdb.number(film.get("vote_count")) >= MIN_VOTES_THRESHOLD:
                ids.append(movie_id)
        if page >= tmdb.number(answer.get("total_pages")):
            break
    return ids[:per_year]


def film_ids(client: Client, per_year: int) -> list[int]:
    """Return the Kaggle ids, then each year's most-voted films, each id once."""
    kaggle_ids = pd.read_csv(KAGGLE_MOVIES_CSV, usecols=["id"])["id"].tolist()
    ids = list(kaggle_ids)
    for year in range(FIRST_YEAR, date.today().year + 1):
        found = discover_year(client, year, per_year)
        print(f"{year}: {len(found)} films")
        ids += found
    # dict.fromkeys drops repeats and keeps the first of each.
    ids = list(dict.fromkeys(ids))
    added = len(ids) - len(kaggle_ids)
    print(f"{len(kaggle_ids):,} Kaggle films and {added:,} more from the yearly lists")
    return ids


def fetch_movies(client: Client, ids: list[int]) -> list[dict]:
    """Return the details answers of ids in order, without the films TMDb no longer has."""

    def fetch(movie_id):
        return client.get(f"/movie/{movie_id}", DETAILS_PARAMS)

    pool = ThreadPoolExecutor(max_workers=WORKERS)
    try:
        answers = list(pool.map(fetch, ids))
    finally:
        # After a failure, the films not started yet are skipped instead of fetched.
        pool.shutdown(cancel_futures=True)
    return [answer for answer in answers if answer is not None]


def is_excluded(movie: dict) -> bool:
    """Return whether the catalogue leaves out a details answer (see EXCLUDED_KEYWORDS)."""
    if movie.get("adult") is True or tmdb.valid_id(movie.get("id")) in EXCLUDED_IDS:
        return True
    keywords = tmdb.records(tmdb.record(movie.get("keywords")).get("keywords"))
    return any(tmdb.text(item.get("name")).casefold() in EXCLUDED_KEYWORDS for item in keywords)


def to_json(items: list[dict], keys: tuple[str, ...]) -> str:
    """Return items as a JSON list column of the Kaggle CSVs, with only the given keys."""
    kept = [{key: item.get(key) for key in keys} for item in items]
    return json.dumps(kept, ensure_ascii=False, separators=(",", ":"))


def turkish_title(movie: dict) -> str:
    """Return the Turkish title of a details answer, or "" when TMDb has none or the same one."""
    title = tmdb.text(movie.get("title"))
    translations = tmdb.records(tmdb.record(movie.get("translations")).get("translations"))
    for translation in translations:
        if translation.get("iso_639_1") == "tr":
            turkish = tmdb.text(tmdb.record(translation.get("data")).get("title")).strip()
            if turkish.casefold() != title.strip().casefold():
                return turkish
            return ""
    return ""


def movie_row(movie: dict) -> dict:
    """Return the movies-CSV row of a details answer with its appended fields.

    The last column, title_tr, is not in the Kaggle files; movierec.dataset reads it when
    it is there.
    """
    keywords = tmdb.records(tmdb.record(movie.get("keywords")).get("keywords"))
    return {
        "id": movie.get("id"),
        "title": tmdb.text(movie.get("title")),
        "overview": tmdb.text(movie.get("overview")),
        "genres": to_json(tmdb.records(movie.get("genres")), ("id", "name")),
        "keywords": to_json(keywords, ("id", "name")),
        "release_date": tmdb.text(movie.get("release_date")),
        "vote_average": tmdb.number(movie.get("vote_average")),
        "vote_count": int(tmdb.number(movie.get("vote_count"))),
        "popularity": tmdb.number(movie.get("popularity")),
        "tagline": tmdb.text(movie.get("tagline")),
        "runtime": int(tmdb.number(movie.get("runtime"))),
        "title_tr": turkish_title(movie),
    }


def credit_row(movie: dict) -> dict:
    """Return the credits-CSV row of a details answer: the top-billed cast and the directors."""
    credits = tmdb.record(movie.get("credits"))
    # TMDb lists the cast in billing order.
    cast = tmdb.records(credits.get("cast"))[:CAST_LIMIT]
    crew = tmdb.records(credits.get("crew"))
    directors = [person for person in crew if person.get("job") == "Director"]
    return {
        "movie_id": movie.get("id"),
        "cast": to_json(cast, ("id", "name", "character")),
        "crew": to_json(directors, ("id", "name", "job")),
    }


def write_atomic(path: Path, text: str) -> None:
    """Write text to a temp file and rename it to path, so a crash never leaves half a file."""
    # The temp name keeps the suffix, so a temp CSV left behind is gitignored too.
    temp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def write_catalogue(out: Path, movies: list[dict], fetched_on: date) -> None:
    """Write the movies and credits CSVs that movierec.dataset reads, and the fetch date."""
    movies_df = pd.DataFrame(
        [movie_row(movie) for movie in movies], columns=[*MOVIE_COLUMNS, "title_tr"]
    )
    credits_df = pd.DataFrame([credit_row(movie) for movie in movies], columns=CREDIT_COLUMNS)
    out.mkdir(parents=True, exist_ok=True)
    write_atomic(out / "tmdb_5000_movies.csv", movies_df.to_csv(index=False, lineterminator="\n"))
    write_atomic(out / "tmdb_5000_credits.csv", credits_df.to_csv(index=False, lineterminator="\n"))
    write_atomic(out / "fetched_on.txt", fetched_on.isoformat() + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a fresh film catalogue from TMDb.")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="folder for the CSVs")
    parser.add_argument(
        "--per-year", type=int, default=PER_YEAR, help="most-voted films taken per year"
    )
    parser.add_argument("--limit", type=int, help="fetch only the first N films, as a trial")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not TMDB_API_KEY:
        print("TMDB_API_KEY is not set; add it to .env.", file=sys.stderr)
        return 1
    client = Client(make_session())
    try:
        ids = film_ids(client, args.per_year)[: args.limit]
        print(f"Fetching {len(ids):,} films...")
        movies = fetch_movies(client, ids)
    except FetchError as exc:
        print(f"TMDb request failed ({exc}); nothing was written.", file=sys.stderr)
        return 1
    kept = [movie for movie in movies if not is_excluded(movie)]
    write_catalogue(args.out, kept, date.today())
    dropped = len(ids) - len(movies)
    print(
        f"Wrote {len(kept):,} films to {args.out}; {dropped:,} dropped (not on TMDb anymore), "
        f"{len(movies) - len(kept):,} left out (EXCLUDED_KEYWORDS)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
