# MovieRec

MovieRec is a content-based movie recommender built with Flask on the [TMDB 5000 dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). It finds films similar to one you like, or ranks films that match a set of filters (genres, director, release years, minimum rating). Movie and person pages are enriched with live data from the TMDb API: posters, trailers, cast and biographies. The interface is in Turkish.

There are no accounts. Favourites and the watchlist are kept in the visitor's browser, so the app keeps no accounts or lists on the server and needs no database. The site follows the system's light or dark setting, and a header button switches it.

![Home page](screenshots/home.jpg)
![Similar movies](screenshots/similar-movies.jpg)
![Movie page](screenshots/movie-page.jpg)
![My lists](screenshots/my-lists.jpg)
![Movie page in the dark theme](screenshots/movie-page-dark.jpg)

## How recommendations work

**Similar movies.** Each film gets four TF-IDF blocks: its genres and its directors as whole names ("Science Fiction" and "Christopher Nolan" are one term each), the words of its keywords, and its overview and tagline as English text (stop words removed, up to 20,000 terms). Each block is L2-normalised and weighted (genres 0.4, directors 0.3, keywords 0.3, text 0.6), and the joined vector is normalised again, so the cosine similarity between two films is a dot product. Separate blocks keep a long overview from drowning out the metadata, so a shared character name counts for less. Only the sparse matrix (about 2.5 MB) is kept in memory; a film's similarity row is computed when it is needed.

For a query film, the 91 most similar films form a candidate pool, and each candidate gets a weighted score in which every signal is scaled to 0-1 within the pool. Popularity and vote count are log-scaled first, so one blockbuster in the pool does not push every other film's score to zero:

| Signal | Weight |
|---|---|
| Content similarity (relative to the best match in the pool) | 0.50 |
| Vote average / 10 | 0.20 |
| log(1 + popularity) | 0.20 |
| log(1 + vote count) | 0.10 |

TMDb ratings are noisy below 100 votes, so such films are dropped whenever the pool contains a film with at least 100 votes. Because the pool size is fixed, asking for more results never reorders the top ones.

**Filter search.** Films that match every selected genre, the director and the year and rating bounds are ranked by `0.40 * rating + 0.45 * popularity + 0.15 * vote count` (each scaled to 0-1). Co-directed films match any of their directors.

**Title search.** Exact, case-insensitive title matches win; otherwise the closest title by fuzzy matching (thefuzz `WRatio`, cutoff 75) is used. When several films share a title, the most-voted one is chosen.

The weights live in [movierec/config.py](movierec/config.py).

## Evaluation

There is no user data to test against, so `scripts/evaluate.py` uses every film in the dataset as a seed and measures how its 12 similar movies agree with it on metadata, how much of the catalogue they reach and how popular they are:

```bash
python scripts/evaluate.py
```

| Method | Genre overlap | Same director | Catalog coverage | Median votes |
|---|---|---|---|---|
| Pure cosine | 0.845 | 10.7% | 98.9% | 206 |
| MovieRec (hybrid) | 0.814 | 8.4% | 58.3% | 1,486 |
| Most-voted baseline | 0.159 | 0.1% | 0.3% | 10,099 |

Before the field-weighted blocks and log scaling, when all fields formed one text, pure cosine scored 0.356 / 4.5% / 98.6% / 284 and the hybrid 0.335 / 4.0% / 64.1% / 1,630.

Toy Story's top 5, before and after:

- Before: Toy Story 2, Toy Story 3, Monsters, Inc., Small Soldiers, The 40 Year Old Virgin (both overviews mention "Andy")
- After: Toy Story 2, Toy Story 3, A Bug's Life, Monsters, Inc., Cars

The median film in the dataset has 235 votes. *Pure cosine* is the top 12 by TF-IDF similarity alone, *MovieRec (hybrid)* is exactly what the app shows, and the *most-voted baseline* recommends the same 12 most-voted films to everyone. Genre overlap is the mean Jaccard index between the seed's and each recommendation's genre sets (seeds without genres are skipped), same director is the share of recommendations that share a director with the seed, catalog coverage is the share of all films recommended at least once, and median votes is the median vote count of all recommendations, counting a film each time it is recommended.

- These are proxies for metadata agreement, not accuracy: they show whether the lists are coherent, not whether people would like them.
- Genres and directors are now blocks of their own in the model, so the first two columns are largely circular: most of the jump in genre overlap (0.335 to 0.814) is the model rewarding exactly what the column counts. Coverage and median votes do not depend on the model's inputs.
- The weights are round, hand-picked values, not a tuned optimum. The director weight started at 0.5 and was lowered to 0.3 after reading the lists: at 0.5, 438 films got 8 or more of their 12 picks from their own director, so prolific directors pushed out sequels (Harry Potter, Jaws). At 0.3 it is 26 films.
- The hybrid gives up a little agreement against pure cosine (genre overlap 0.845 to 0.814) and over a third of the catalogue for better-known films: its picks have a median of 1,486 votes against 206 for pure cosine. The log scaling lowers that median on purpose (1,660 with linear scaling), so the biggest blockbusters no longer decide the order.
- Catalog coverage got worse, 64.1% to 58.3%, because of the new blocks. Most of the gap to pure cosine comes from the 100-vote floor: only 3,162 films (65.8%) have 100 votes or more, and the hybrid reaches 88% of them (97% before).

## Project layout

```text
app.py                  entry point (gunicorn app:app)
movierec/
  config.py             settings read from the environment / .env
  dataset.py            CSV loading and preprocessing
  similarity.py         TF-IDF matrix and ranking formulas
  engine.py             RecommendationEngine: similar movies and filter search, offline
  search_filters.py     input parsing and genre/director filters
  movie_details.py      movie pages and list cards enriched with TMDb data
  genres.py             genre names and Turkish labels
  tmdb.py               small TMDb API client
  people.py             director/actor pages from TMDb
  web/
    __init__.py         app factory: config, security headers, error pages
    movies.py           home, movie, person and search routes, JSON suggestions
    lists.py            favourites and watchlist page, card endpoint
    redirects.py        safe redirects back to the referring page
templates/              Jinja templates
static/                 CSS, JavaScript and the icon sprite; js/list_store.js keeps the lists in localStorage
scripts/evaluate.py     offline evaluation of the similar-movies lists
tests/                  pytest suite; tests/js/ holds the node:test suite
```

## Running locally

Requirements: Python 3.12 and the two Kaggle CSVs.

1. Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) into `dataset/`.
2. Create the environment file and fill in a [TMDb API key](https://www.themoviedb.org/settings/api) and a secret key:

   ```bash
   cp .env.example .env
   python -c "import secrets; print(secrets.token_hex(32))"   # value for FLASK_SECRET_KEY
   ```

   Without a TMDb key the app still works on the dataset alone, without posters, trailers or person pages. The secret key signs the cookie that carries flash messages; without it a random key is generated on each start.
3. Install and run:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python app.py
   ```

   The app listens on http://127.0.0.1:5001.

With Docker, put the CSVs in `dataset/` first; they are copied into the image. Then `docker compose up --build` serves the app with gunicorn on port 5001, as a non-root user with a health check.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest
node --test "tests/js/*.test.mjs"
```

Without the Kaggle CSVs, pytest runs against a 158-film fixture taken from the dataset (`tests/fixtures/`) and only skips the engine output snapshot, which needs all 4,803 films. CI runs the same commands on every push.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `TMDB_API_KEY` | TMDb v3 API key | unset (offline mode) |
| `FLASK_SECRET_KEY` | signs the flash-message cookie | random per process |
| `DATASET_DIR` | folder with the two CSVs | `dataset/` |
| `SESSION_COOKIE_SECURE` | set to `1` behind HTTPS | off |
| `HOST`, `PORT`, `FLASK_DEBUG` | development server options | `127.0.0.1`, `5001`, off |
| `LOG_LEVEL` | log level, also under gunicorn | `INFO` |

## Limitations

- The dataset ends in 2017, and recommendations only come from its 4,803 films.
- Recommendations are purely content-based; they do not learn from what users like.
- Lists need JavaScript, belong to one browser on one device and disappear when its site data is cleared.

## License and credits

Source available under the [PolyForm Noncommercial License 1.0.0](LICENSE): free for personal, educational and other noncommercial use; commercial use is not permitted.

Movie data and images come from [TMDB](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB. The base dataset is the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) on Kaggle.

The icons in `static/icons.svg` are from [Font Awesome Free 6.5.1](https://fontawesome.com) and stay under their own [CC BY 4.0 license](https://fontawesome.com/license/free). The type is [IBM Plex Sans and IBM Plex Sans Condensed](https://github.com/IBM/plex) (SIL Open Font License 1.1), loaded from Google Fonts.

## Türkçe özet

MovieRec, TMDB 5000 veri seti üzerinde çalışan, Flask ile yazılmış içerik tabanlı bir film öneri sistemidir. Beğendiğiniz bir filme benzeyen filmleri ya da seçtiğiniz tür, yönetmen, yıl ve puan filtrelerine uyan filmleri sıralar. Benzerlik, filmlerin tür, anahtar kelime, yönetmen, özet ve slogan metinlerinden üretilen TF-IDF vektörleriyle hesaplanır; sonuçlar puan, popülerlik ve oy sayısıyla birlikte ağırlıklandırılır. Benzer film listeleri, kullanıcı verisi olmadan `scripts/evaluate.py` ile meta veri üzerinden değerlendirilir: tür örtüşmesi, aynı yönetmen oranı, katalog kapsamı ve önerilen filmlerin popülerliği.

Üyelik yoktur: beğeniler ve izleme listesi ziyaretçinin tarayıcısında saklanır, bu yüzden veritabanı gerekmez. Proje ticari olmayan kullanım için PolyForm Noncommercial 1.0.0 lisansıyla paylaşılmıştır. Kurulum adımları yukarıdaki bölümlerdedir.
