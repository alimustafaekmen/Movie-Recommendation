"""scripts/fetch_catalogue.py against a fake TMDb that gives hand-written, fictional answers."""

import json
import logging
import math
import threading
import time
import traceback
from datetime import date

import pandas as pd
import pytest
import requests
from conftest import FAKE_TMDB_KEY, REPO, FakeResponse, FakeSession

from movierec.config import MIN_VOTES_THRESHOLD, TMDB_API_BASE_URL
from movierec.dataset import CAST_LIMIT, CREDIT_COLUMNS, MOVIE_COLUMNS, load_movies
from scripts import fetch_catalogue

CAST = [
    {"id": 900 + n, "name": f"Actor {n}", "character": f"Role {n}", "order": n, "gender": 0}
    for n in range(20)
]
TIDEKEEPER = {
    "id": 101,
    "title": "The Tidekeeper's Clock",
    "overview": 'A keeper finds a clock that runs "backwards", one tide at a time.\nThen it stops.',
    "tagline": "Time turns with the tide.",
    "release_date": "2021-03-04",
    "runtime": 118,
    "vote_average": 7.4,
    "vote_count": 1520,
    "popularity": 33.5,
    "adult": False,
    "genres": [{"id": 18, "name": "Drama"}, {"id": 878, "name": "Science Fiction"}],
    "keywords": {"keywords": [{"id": 1, "name": "lighthouse"}, {"id": 2, "name": "time loop"}]},
    "credits": {
        "cast": CAST,
        "crew": [
            {"id": 1, "name": "Mira Holt", "job": "Director", "department": "Directing"},
            {"id": 2, "name": "Ode Vance", "job": "Screenplay", "department": "Writing"},
            {"id": 3, "name": "Sam Reyes", "job": "Director", "department": "Directing"},
        ],
    },
    "translations": {
        "translations": [
            {"iso_639_1": "de", "data": {"title": "Die Uhr des Gezeitenwächters"}},
            {"iso_639_1": "tr", "data": {"title": "Gelgit Bekçisinin Saati", "overview": ""}},
        ]
    },
}
# A new film whose page on TMDb is still mostly empty.
PAPER_MOONS = {
    "id": 102,
    "title": "Paper Moons",
    "overview": "",
    "tagline": None,
    "release_date": "",
    "runtime": None,
    "vote_average": 0,
    "vote_count": 0,
    "popularity": 1.2,
    "genres": [],
    "keywords": {"keywords": []},
    "credits": {"cast": [], "crew": []},
    # TMDb gives its Turkish title the same as the English one.
    "translations": {"translations": [{"iso_639_1": "tr", "data": {"title": " paper MOONS "}}]},
}


class CatalogueSession(FakeSession):
    """A FakeSession that also answers /discover/movie, 20 films a page, from films_by_year."""

    def __init__(self, routes=None, films_by_year=None):
        super().__init__(routes)
        self.films_by_year = films_by_year or {}

    def get(self, url, params=None, timeout=None):
        if url != TMDB_API_BASE_URL + "/discover/movie":
            return super().get(url, params, timeout)
        self.calls.append(("/discover/movie", dict(params)))
        films = self.films_by_year.get(int(params["primary_release_date.gte"][:4]), [])
        page = params["page"]
        return FakeResponse(
            {
                "results": films[(page - 1) * 20 : page * 20],
                "total_pages": math.ceil(len(films) / 20),
            }
        )


@pytest.fixture(autouse=True)
def offline_script(monkeypatch, tmp_path):
    """The script with the fake key, no rate cap to wait for, and three Kaggle seed ids."""
    seeds = tmp_path / "kaggle_movies.csv"
    pd.DataFrame({"id": [101, 404, 102], "title": ["A", "B", "C"]}).to_csv(seeds, index=False)
    monkeypatch.setattr(fetch_catalogue, "KAGGLE_MOVIES_CSV", seeds)
    monkeypatch.setattr(fetch_catalogue, "TMDB_API_KEY", FAKE_TMDB_KEY)
    monkeypatch.setattr(fetch_catalogue, "REQUESTS_PER_SECOND", 10_000)


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Run the script on a fake session, writing to tmp_path / "out"; returns the exit code."""

    def run_script(session, *args):
        monkeypatch.setattr(fetch_catalogue, "make_session", lambda: session)
        return fetch_catalogue.main(["--out", str(tmp_path / "out"), *args])

    return run_script


def test_defaults_take_400_films_a_year_into_the_gitignored_folder():
    args = fetch_catalogue.parse_args([])

    assert args.out == REPO / "dataset" / "refreshed"
    assert args.per_year == 400
    assert args.limit is None


def test_a_details_answer_becomes_rows_of_the_kaggle_csvs():
    movie = fetch_catalogue.movie_row(TIDEKEEPER)
    credit = fetch_catalogue.credit_row(TIDEKEEPER)

    assert list(movie) == [*MOVIE_COLUMNS, "title_tr"]
    assert list(credit) == CREDIT_COLUMNS
    assert movie["id"] == credit["movie_id"] == 101
    # Kaggle stores genres, keywords, cast and crew as JSON lists of objects.
    assert json.loads(movie["genres"]) == TIDEKEEPER["genres"]
    assert json.loads(movie["keywords"]) == TIDEKEEPER["keywords"]["keywords"]
    assert json.loads(credit["cast"]) == [
        {"id": person["id"], "name": person["name"], "character": person["character"]}
        for person in CAST[:CAST_LIMIT]
    ]
    # Of the crew, only the directors are kept.
    assert json.loads(credit["crew"]) == [
        {"id": 1, "name": "Mira Holt", "job": "Director"},
        {"id": 3, "name": "Sam Reyes", "job": "Director"},
    ]
    assert movie["title_tr"] == "Gelgit Bekçisinin Saati"


@pytest.mark.parametrize(
    "translations",
    [
        None,
        {"translations": []},
        {"translations": [{"iso_639_1": "de", "data": {"title": "Die Uhr"}}]},
        {"translations": [{"iso_639_1": "tr", "data": {"title": ""}}]},
        {"translations": [{"iso_639_1": "tr", "data": None}]},
        {"translations": [{"iso_639_1": "tr", "data": {"title": "The Tidekeeper's Clock "}}]},
        {"translations": "broken"},
    ],
)
def test_the_turkish_title_is_kept_only_when_tmdb_has_a_different_one(translations):
    movie = {**TIDEKEEPER, "translations": translations}

    assert fetch_catalogue.movie_row(movie)["title_tr"] == ""


def test_the_written_catalogue_loads_like_the_kaggle_files(tmp_path):
    out = tmp_path / "out"
    fetch_catalogue.write_catalogue(out, [TIDEKEEPER, PAPER_MOONS], date(2026, 10, 22))
    df = load_movies(str(out / "tmdb_5000_movies.csv"), str(out / "tmdb_5000_credits.csv"))
    movies = df.set_index("id")

    assert list(movies.index) == [101, 102]
    film = movies.loc[101]
    assert film["title"] == "The Tidekeeper's Clock"
    assert film["title_tr"] == "Gelgit Bekçisinin Saati"
    assert film["overview"] == TIDEKEEPER["overview"]
    assert film["tagline"] == "Time turns with the tide."
    assert film["genre_list"] == ["Drama", "Science Fiction"]
    assert film["genres"] == "Drama Science Fiction"
    assert film["keywords"] == "lighthouse time loop"
    assert film["directors"] == ["Mira Holt", "Sam Reyes"]
    assert film["cast"] == ", ".join(f"Actor {n}" for n in range(CAST_LIMIT))
    assert film["year"] == 2021
    assert (film["vote_average"], film["vote_count"]) == (7.4, 1520)
    assert (film["popularity"], film["runtime"]) == (33.5, 118)
    sparse = movies.loc[102]
    assert (sparse["year"], sparse["runtime"], sparse["tagline"]) == (0, 0, "")
    assert sparse["title_tr"] == ""
    assert (sparse["genre_list"], sparse["directors"], sparse["cast"]) == ([], [], "")
    assert (out / "fetched_on.txt").read_text() == "2026-10-22\n"
    # Every file was renamed into place; no temp file is left.
    assert sorted(path.name for path in out.iterdir()) == [
        "fetched_on.txt",
        "tmdb_5000_credits.csv",
        "tmdb_5000_movies.csv",
    ]


def test_discover_takes_a_years_most_voted_films_page_by_page():
    films = [{"id": n, "vote_count": 5000 - n} for n in range(1, 46)]
    session = CatalogueSession(films_by_year={2020: films})

    ids = fetch_catalogue.discover_year(fetch_catalogue.Client(session), 2020, per_year=30)

    assert ids == list(range(1, 31))
    # 30 films fill two pages of 20; the third page is never asked for.
    assert [params["page"] for _, params in session.calls] == [1, 2]
    params = session.calls[0][1]
    assert params["sort_by"] == "vote_count.desc"
    assert params["primary_release_date.gte"] == "2020-01-01"
    assert params["primary_release_date.lte"] == "2020-12-31"
    assert params["vote_count.gte"] == MIN_VOTES_THRESHOLD
    assert params["api_key"] == FAKE_TMDB_KEY


def test_discover_keeps_to_the_vote_floor_and_stops_at_the_last_page():
    # An answer that ignores vote_count.gte still yields only films above the floor.
    films = [
        {"id": 1, "vote_count": 300},
        {"id": 2, "vote_count": MIN_VOTES_THRESHOLD},
        {"id": 3, "vote_count": MIN_VOTES_THRESHOLD - 1},
    ]
    session = CatalogueSession(films_by_year={2026: films})

    ids = fetch_catalogue.discover_year(fetch_catalogue.Client(session), 2026, per_year=400)

    assert ids == [1, 2]
    assert len(session.calls) == 1


def test_the_film_list_is_the_kaggle_ids_then_each_years_new_films_once():
    # The seeds are 101, 404 and 102; 102 is also among 1990's most-voted films.
    session = CatalogueSession(
        films_by_year={
            1990: [{"id": 102, "vote_count": 900}, {"id": 201, "vote_count": 800}],
            1991: [{"id": 202, "vote_count": 700}],
        }
    )

    ids = fetch_catalogue.film_ids(fetch_catalogue.Client(session), per_year=5)

    assert ids == [101, 404, 102, 201, 202]
    years = [int(params["primary_release_date.gte"][:4]) for _, params in session.calls]
    assert years == list(range(1990, date.today().year + 1))


def test_a_run_fetches_each_film_once_and_drops_the_ones_tmdb_no_longer_has(run, tmp_path, capsys):
    session = CatalogueSession(
        routes={
            "/movie/101": FakeResponse(TIDEKEEPER),
            "/movie/102": FakeResponse(PAPER_MOONS),
            "/movie/103": FakeResponse({"id": 103, "title": "Glass Orchard"}),
        },
        # 404 is gone from TMDb; 104 is past the cap of two films a year.
        films_by_year={
            2020: [
                {"id": 103, "vote_count": 900},
                {"id": 101, "vote_count": 800},
                {"id": 104, "vote_count": 700},
            ]
        },
    )

    assert run(session, "--per-year", "2") == 0

    movie_calls = [(path, params) for path, params in session.calls if path.startswith("/movie/")]
    assert sorted(path for path, _ in movie_calls) == [
        "/movie/101",
        "/movie/102",
        "/movie/103",
        "/movie/404",
    ]
    for _, params in movie_calls:
        assert params["language"] == "en-US"
        assert params["append_to_response"] == "credits,keywords,translations"
    out = tmp_path / "out"
    df = load_movies(str(out / "tmdb_5000_movies.csv"), str(out / "tmdb_5000_credits.csv"))
    assert df["id"].tolist() == [101, 102, 103]
    assert (out / "fetched_on.txt").read_text() == date.today().isoformat() + "\n"
    summary = capsys.readouterr().out
    assert "Wrote 3 films" in summary
    assert "1 dropped" in summary


def keywords(*names):
    return {"keywords": {"keywords": [{"id": n, "name": name} for n, name in enumerate(names)]}}


@pytest.mark.parametrize(
    "change",
    [
        keywords("lighthouse", "Erotic Movie"),
        keywords("softcore"),
        keywords("torture porn"),
        {"adult": True},
        {"id": min(fetch_catalogue.EXCLUDED_IDS)},
    ],
)
def test_erotic_shock_and_adult_films_are_left_out(change):
    assert fetch_catalogue.is_excluded({**TIDEKEEPER, **change})


def test_a_dark_subject_alone_leaves_no_film_out():
    # Such tags also mark well-known dramas, so only the tags of the genres themselves count.
    dark = keywords("rape", "pornography", "porn industry", "eroticism", "prostitute")

    assert not fetch_catalogue.is_excluded({**TIDEKEEPER, **dark})
    assert not fetch_catalogue.is_excluded(TIDEKEEPER)
    assert not fetch_catalogue.is_excluded(PAPER_MOONS)
    assert not fetch_catalogue.is_excluded({"id": 105, "keywords": None})


def test_a_run_leaves_the_excluded_films_out_and_counts_them(run, tmp_path, capsys):
    session = CatalogueSession(
        routes={
            "/movie/101": FakeResponse(TIDEKEEPER),
            "/movie/102": FakeResponse({**PAPER_MOONS, **keywords("sexploitation")}),
        }
    )

    assert run(session) == 0

    out = tmp_path / "out"
    df = load_movies(str(out / "tmdb_5000_movies.csv"), str(out / "tmdb_5000_credits.csv"))
    assert df["id"].tolist() == [101]
    summary = capsys.readouterr().out
    assert "Wrote 1 films" in summary
    assert "1 left out" in summary


def test_a_trial_run_fetches_only_the_first_films(run, tmp_path):
    session = CatalogueSession(routes={"/movie/101": FakeResponse(TIDEKEEPER)})

    assert run(session, "--limit", "2") == 0

    assert sorted(path for path in session.paths if path.startswith("/movie/")) == [
        "/movie/101",
        "/movie/404",
    ]
    out = tmp_path / "out"
    df = load_movies(str(out / "tmdb_5000_movies.csv"), str(out / "tmdb_5000_credits.csv"))
    assert df["id"].tolist() == [101]


def test_a_failed_request_exits_non_zero_and_writes_nothing(run, tmp_path, capsys):
    session = CatalogueSession(
        routes={"/movie/101": FakeResponse(TIDEKEEPER), "/movie/102": FakeResponse(status_code=500)}
    )

    assert run(session) == 1

    assert not (tmp_path / "out").exists()
    error = capsys.readouterr().err
    assert "/movie/102" in error
    assert "HTTP 500" in error


class BrokenConnectionSession(FakeSession):
    """Logs a retry the way urllib3 does, URL and key included, then fails the request."""

    def get(self, url, params=None, timeout=None):
        full_url = f"{url}?api_key={params['api_key']}"
        logging.getLogger("urllib3.connectionpool").warning("Retrying after: %s", full_url)
        raise requests.ConnectionError(f"Max retries exceeded with url: {full_url}")


def test_the_key_never_shows_in_errors_or_logs(run, capsys, caplog):
    with caplog.at_level(logging.DEBUG):
        code = run(BrokenConnectionSession())

    assert code == 1
    output = capsys.readouterr()
    assert "/discover/movie" in output.err
    assert "ConnectionError" in output.err
    assert FAKE_TMDB_KEY not in output.out + output.err
    assert FAKE_TMDB_KEY not in caplog.text


def test_a_failed_request_raises_without_the_key_in_its_traceback():
    client = fetch_catalogue.Client(BrokenConnectionSession())

    with pytest.raises(fetch_catalogue.FetchError) as excinfo:
        client.get("/movie/1", {})

    assert FAKE_TMDB_KEY not in "".join(traceback.format_exception(excinfo.value))


def test_without_a_key_the_script_stops_before_any_request(run, monkeypatch, capsys):
    monkeypatch.setattr(fetch_catalogue, "TMDB_API_KEY", None)
    session = CatalogueSession()

    assert run(session) == 1

    assert session.calls == []
    assert "TMDB_API_KEY" in capsys.readouterr().err


def test_the_session_retries_rate_limits_and_server_errors():
    adapter = fetch_catalogue.make_session().get_adapter(TMDB_API_BASE_URL)
    retry = adapter.max_retries

    assert retry.total >= 3
    assert retry.backoff_factor > 0
    assert {429, 500, 502, 503, 504} <= set(retry.status_forcelist)
    # A 429 answer says in Retry-After how long to wait.
    assert retry.respect_retry_after_header
    assert adapter.poolmanager.connection_pool_kw["maxsize"] >= fetch_catalogue.WORKERS


def test_requests_from_all_threads_share_one_rate_cap(monkeypatch):
    monkeypatch.setattr(fetch_catalogue, "REQUESTS_PER_SECOND", 100)
    client = fetch_catalogue.Client(FakeSession({"/movie/1": FakeResponse({"id": 1})}))

    def five_requests():
        for _ in range(5):
            client.get("/movie/1", {})

    threads = [threading.Thread(target=five_requests) for _ in range(4)]
    start = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # 20 requests at 100 a second: the last one starts at least 0.19 s after the first.
    assert time.monotonic() - start >= 0.18
