import os
from pathlib import Path

import pytest
import requests
from pages import body_text

REPO = Path(__file__).resolve().parents[1]
KAGGLE_DATASET_DIR = REPO / "dataset"
FIXTURE_DATASET_DIR = REPO / "tests" / "fixtures" / "dataset"
KAGGLE_CSVS = [
    KAGGLE_DATASET_DIR / name for name in ("tmdb_5000_movies.csv", "tmdb_5000_credits.csv")
]

# Without the Kaggle CSVs (a fresh clone, CI) the suite runs on a small fixture dataset;
# USE_FIXTURE_DATASET=1 forces that locally. Tests marked real_dataset need the full CSVs.
USE_FIXTURE_DATASET = os.getenv("USE_FIXTURE_DATASET") == "1" or not all(
    p.exists() for p in KAGGLE_CSVS
)
DATASET_DIR = FIXTURE_DATASET_DIR if USE_FIXTURE_DATASET else KAGGLE_DATASET_DIR

# These must be set before anything imports movierec.config. load_dotenv() never overrides
# variables that already exist, so the suite runs offline and deterministically even when
# .env holds a real key or points DATASET_DIR at another catalogue: every TMDb call is
# skipped while the key is empty, and the tests read only the dataset picked above.
os.environ["TMDB_API_KEY"] = ""
os.environ["FLASK_SECRET_KEY"] = "test-secret-key"
os.environ["DATASET_DIR"] = str(DATASET_DIR)

# The key the fake TMDb session expects; tests check that it never leaks into logs.
FAKE_TMDB_KEY = "SECRET123"

# What movierec.people.find_person returns, for tests that stub it.
PERSON = {
    "id": 3894,
    "name": "Christian Bale",
    "biography": "Christian Bale is an English actor.",
    "birthday": "1974-01-30",
    "deathday": None,
    "place_of_birth": "Haverfordwest, Wales, UK",
    "known_for_department": "Acting",
    "profile_image_url": "https://image.tmdb.org/t/p/h632/profile.jpg",
    "homepage": None,
    "movies": [
        {
            "id": 155,
            "title": "The Dark Knight",
            "poster_url": "https://image.tmdb.org/t/p/w342/poster.jpg",
            "year": 2008,
            "vote_average": 8.2,
            "character": "Bruce Wayne",
        }
    ],
    "person_type_tr": "Oyuncu",
}

# Home page searches, each a shareable GET URL: results, a warning and the empty state.
SEARCHES = {
    "results": "/?film=Avatar&n=5",
    "warning": "/?film=Avatr&n=5",
    "empty": "/?tur=Documentary&min_puan=8.5&min_yil=1900&max_yil=1905",
}


def pytest_collection_modifyitems(items):
    if not USE_FIXTURE_DATASET:
        return
    skip = pytest.mark.skip(reason="needs the full Kaggle dataset")
    for item in items:
        if "real_dataset" in item.keywords:
            item.add_marker(skip)


def _refuse_network(*args, **kwargs):
    raise RuntimeError("network disabled in tests")


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    # requests.get() and friends all end up in Session.request.
    monkeypatch.setattr(requests.sessions.Session, "request", _refuse_network)


@pytest.fixture(scope="session")
def app_module():
    # Session fixtures are set up before the function-scoped block_network, so the
    # import that builds the engine needs its own guard.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(requests.sessions.Session, "request", _refuse_network)
        import app
    return app


@pytest.fixture
def client(app_module):
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


@pytest.fixture(scope="session")
def engine(app_module):
    return app_module.app.extensions["engine"]


@pytest.fixture(scope="session")
def details(app_module):
    return app_module.app.extensions["details"]


@pytest.fixture
def person_client(client, monkeypatch):
    monkeypatch.setattr("movierec.web.movies.find_person", lambda name, person_type: dict(PERSON))
    return client


def _pages(client, person_client):
    """Every page type once."""
    return {
        "home": client.get("/"),
        "genre": client.get("/?tur=Action"),
        "results": client.get(SEARCHES["results"]),
        "movie": client.get("/movie/155"),
        "person": person_client.get("/actor/Christian+Bale"),
        "my_lists": client.get("/my_lists"),
        "not_found": client.get("/does-not-exist"),
    }


@pytest.fixture
def pages(client, person_client):
    return {name: body_text(response) for name, response in _pages(client, person_client).items()}


@pytest.fixture
def cold_details(details, monkeypatch):
    """The shared MovieDetails with an empty poster cache, restored after the test."""
    monkeypatch.setattr(details, "poster_cache", {})
    return details


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload if payload is not None else {}
        self.status_code = status_code

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self.payload


class FakeSession:
    """Answers TMDb paths from a dict; a list value is consumed one item per call."""

    def __init__(self, routes=None):
        self.routes = dict(routes or {})
        self.calls = []

    @property
    def paths(self):
        return [path for path, _ in self.calls]

    def get(self, url, params=None, timeout=None):
        from movierec import tmdb

        path = url.removeprefix(tmdb.TMDB_API_BASE_URL)
        self.calls.append((path, dict(params or {})))
        answer = self.routes.get(path, FakeResponse(status_code=404))
        if isinstance(answer, list):
            answer = answer.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def fake(monkeypatch):
    """A configured TMDb client whose requests go to a FakeSession; unknown paths are 404."""
    from movierec import people, tmdb

    session = FakeSession()
    # Person lookups are cached in memory; each test starts without one.
    monkeypatch.setattr(people, "_person_cache", {})
    monkeypatch.setattr(tmdb, "_session", session)
    monkeypatch.setattr(tmdb, "TMDB_API_KEY", FAKE_TMDB_KEY)
    monkeypatch.setattr(tmdb, "_cooldown_until", 0.0)
    return session
