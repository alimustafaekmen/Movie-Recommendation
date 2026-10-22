# MovieRec

[![CI](https://github.com/alimustafaekmen/Movie-Recommendation/actions/workflows/ci.yml/badge.svg)](https://github.com/alimustafaekmen/Movie-Recommendation/actions/workflows/ci.yml)

MovieRec is a content-based movie recommender built with Flask on the [TMDB 5000 dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). It finds films similar to one you like, or ranks films that match a set of filters (genres, director, release years, minimum rating). Movie and person pages are enriched with live data from the TMDb API: posters, trailers, cast and biographies. The interface is in Turkish.

There are no accounts. Favourites and the watchlist are kept in the visitor's browser, so the app keeps no accounts or lists on the server and needs no database. The lists page also recommends films based on the favourites. The site follows the system's light or dark setting, and a header button switches it.

![Home page](screenshots/home.jpg)
![Similar movies](screenshots/similar-movies.jpg)
![Movie page](screenshots/movie-page.jpg)
![My lists](screenshots/my-lists.jpg)
![Movie page in the dark theme](screenshots/movie-page-dark.jpg)

## How recommendations work

**Similar movies.** Each film becomes four TF-IDF blocks:

- its genres and its directors, as whole names ("Science Fiction" and "Christopher Nolan" are one term each; co-directors stay separate),
- the words of its keywords (over half of the keyword phrases belong to a single film, but words such as "toy" are shared),
- its overview and tagline as English text (stop words removed, at most 20,000 terms).

Each block is L2-normalised and multiplied by its weight (genres 0.3, directors 0.2, keywords 0.4, text 0.6), and the joined row is normalised again. The cosine between two films is then a weighted mean of the four block cosines, where each block counts with the square of its weight. For a film with all four fields that is text 55%, keywords 25%, genres 14% and directors 6%.

Separate blocks cap how much the overview can decide. In the earlier model all fields formed one text, so a rare character name could link unrelated films: Toy Story's top 5 included *The 40 Year Old Virgin* because both overviews mention "Andy".

- Before: Toy Story 2, Toy Story 3, Monsters, Inc., Small Soldiers, The 40 Year Old Virgin
- After: Toy Story 3, Toy Story 2, Monsters, Inc., Big Hero 6, Inside Out

The model is a sparse 4,803 × 29,665 matrix of about 2.5 MB. No film-by-film similarity table is stored; a film's similarities are one sparse product, computed when needed.

**Re-ranking.** The 91 films with the highest cosine (three times the longest list of 30, plus one) form a candidate pool. Each candidate gets a weighted score:

| Signal | Scaled as | Weight |
|---|---|---|
| Content similarity | divided by the best match in the pool | 0.50 |
| Vote average | divided by 10 | 0.20 |
| Popularity | log(1 + x), divided by the pool's maximum | 0.20 |
| Vote count | log(1 + x), divided by the pool's maximum | 0.10 |

Raw cosines in a pool sit close together, so similarity is divided by the best match to use the full 0-1 range. Popularity and vote counts are heavy-tailed (popularity has a median of 13 and a maximum of 876), so they are log-scaled; scaled linearly, one blockbuster would push every other film's term close to 0. TMDb ratings are noisy below 100 votes, so such films are dropped whenever the pool has a film with at least 100 votes. Because the pool size is fixed, asking for more results never reorders the top ones.

**How the weights were chosen.** They are round, hand-picked values; there is no user data to tune against. They were checked by reading the lists of well-known films and with the columns of `scripts/evaluate.py`, including two that the model does not use (shared lead actor and series recall). The director weight started at 0.5, but then 438 films got 8 or more of their 12 picks from their own director, and Jaws's top 8 was all Spielberg. At 0.3, with genres at 0.4, that fell to 26, but same-genre lookalikes pushed out films with the same theme (Seven Samurai got boxing films), Inception and Scarface ranked above the earlier Batman films for The Dark Knight, and series recall dropped from 75.4% to 70.9%. With directors at 0.2, genres at 0.3 and keywords at 0.4 it is 1 film, the Batman films are back at #3-5, Seven Samurai's boxing films drop from #1-2 to #4-5, and series recall is 75.1%, about the earlier model's level. The weights live in [movierec/config.py](movierec/config.py).

**Based on your favourites.** The lists page sends the favourite and watchlist ids from the browser, and the server combines the favourites' similar-movie lists (the first 20 favourites from the dataset, newest first, 30 films each). A film at rank r (counting from 0) in a list scores 1 / (r + 1), and the scores add up over the lists, so a film that several favourites share beats one that only tops a single list. Films already on either list are left out, ties keep the order they first appear in, and the top 12 are shown. Nothing is stored on the server, so no account is needed.

**Filter search.** Films that match every selected genre, the director and the year and rating bounds are ranked by `0.40 * rating + 0.45 * popularity + 0.15 * vote count`. The rating is divided by 10; popularity and vote count are divided by the largest value among the matches. Films under 100 votes are left out unless nothing else matches. Co-directed films match any of their directors.

**Title search.** Exact, case-insensitive title matches win; otherwise the closest title by fuzzy matching (thefuzz `WRatio`, cutoff 75) is used. A title under half the query's length counts only when the query holds all its words, so "Godfather 2" finds *The Godfather*, not *Her*, and tied scores go to the title closest as a whole (`token_sort_ratio`). When several films share a title, the most-voted one is chosen.

## Evaluation

There is no user data to test against, so `scripts/evaluate.py` uses every film as a seed and checks its 12 similar movies. It asks whether they agree with the seed on metadata, whether they find its series, how much of the catalogue they reach and how popular they are.

```bash
python scripts/evaluate.py
```

| Method | Genre overlap | Same director | Shared lead actor | Series recall | Catalog coverage | Median votes |
|---|---|---|---|---|---|---|
| Pure cosine | 0.744 | 5.6% | 3.48% | 76.7% | 98.5% | 191 |
| MovieRec (hybrid) | 0.739 | 5.0% | 3.55% | 75.1% | 61.7% | 1,350 |
| Most-voted baseline | 0.159 | 0.1% | 0.75% | 1.1% | 0.3% | 10,099 |
| Earlier model, pure cosine | 0.356 | 4.5% | 3.23% | 76.4% | 98.6% | 284 |
| Earlier model, hybrid | 0.335 | 4.0% | 3.15% | 75.4% | 64.1% | 1,630 |

*Pure cosine* is the top 12 by content similarity alone, *MovieRec (hybrid)* is what the app shows, and the *most-voted baseline* gives everyone the same 12 most-voted films. The earlier model (commit dc815a0) used one TF-IDF over all fields and linear popularity; its rows come from the same metric code run on that commit's engine.

- **Genre overlap:** the mean Jaccard index of the seed's and the recommendation's genres (seeds without genres are skipped).
- **Same director:** the share of recommendations that share a director with the seed.
- **Shared lead actor:** the share whose three top-billed actors include one of the seed's three. Cast is not a model input.
- **Series recall:** of the seed's same-series films (matched by title stem, at least 100 votes), the share found in the 12, averaged over the 464 films that have such siblings. About 1 title group in 10 is not a real series (Kung Fu Hustle and Kung Fu Panda).
- **Catalog coverage:** the share of all films recommended at least once.
- **Median votes:** the median vote count of all recommendations, counting a film each time it is recommended (the dataset median is 235).

What the numbers show, and what they do not:

- They measure coherence, not accuracy: whether the lists hang together, not whether people would like them.
- Genres and directors are model inputs, so the first two columns are largely circular. With the genre weight set to 0, the hybrid's genre overlap is 0.312. So the jump from 0.335 mostly shows the model doing what it was told.
- Shared lead actor rises with the director weight, because directors reuse actors. Without same-director pairs, every model scores about 2.0-2.2%, so this column does not show the new model is better. Series recall is partly circular too, because character names in overviews link sequels.
- The hybrid trades a little agreement and about 37 points of coverage for better-known films (median 1,350 votes against 191). Log scaling lowers that median on purpose (1,553 with linear scaling).
- Most of the coverage gap comes from the 100-vote floor. Only 3,162 films (65.8%) have 100 votes or more, and the hybrid reaches 94% of them.

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
    lists.py            favourites and watchlist page, card and recommendation endpoints
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

   Open http://localhost:5001. The terminal shows one line per page request. The dev server only
   listens on this computer (`HOST=127.0.0.1`); port 5001 rather than Flask's 5000 because macOS
   uses 5000 for AirPlay Receiver. `PORT=8000 python app.py` picks another port.

With Docker, put the CSVs in `dataset/` first; they are copied into the image. Then `docker compose up --build` serves the app with gunicorn at http://localhost:5001, as a non-root user with a health check. Compose publishes the port on 127.0.0.1 only, keeps the file system read-only (only `/tmp` is writable, in memory, for gunicorn's worker heartbeat) and drops all Linux capabilities.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest --cov
node --test "tests/js/*.test.mjs"
```

What the suites protect:

- **Engine output**: a snapshot of matches and similar-movie lists on the full dataset. After an intended ranking change, run `python tests/snapshot_engine.py --update` and review the diff.
- **TMDb failures**: after a timeout, server error or malformed JSON the page degrades after one call, without a key no call is made, and the key never reaches the logs.
- **Security**: the CSP and other headers on every response, request body and query length limits, and redirects that stay on the site.
- **Lists**: the browser list store and the lists page, with `node:test`.
- **Evaluation**: the metric helpers of `scripts/evaluate.py`.

Without the Kaggle CSVs, pytest runs against a 158-film fixture taken from the dataset (`tests/fixtures/`) and skips the tests that need all 4,803 films: the snapshot and a few ranking checks on real titles. CI runs the same commands on every push and fails below 95% line and branch coverage of `movierec/` and `app.py` (the fixture run measures about 97%).

## Configuration

Set these in `.env` (`.env.example` lists them) or in the environment, which takes precedence.

| Variable | Purpose | Default |
|---|---|---|
| `TMDB_API_KEY` | TMDb v3 API key | unset (offline mode) |
| `FLASK_SECRET_KEY` | signs the flash-message cookie | random per process |
| `DATASET_DIR` | folder with the two CSVs | `dataset/` |
| `SESSION_COOKIE_SECURE` | set to `1` behind HTTPS: Secure cookie and HSTS | off |
| `HOST`, `PORT`, `FLASK_DEBUG` | development server options | `127.0.0.1`, `5001`, off |
| `LOG_LEVEL` | log level, also under gunicorn | `INFO` |

## Limitations

- The dataset ends in 2017, and recommendations only come from its 4,803 films.
- Recommendations are purely content-based and do not learn from users. Without user data the weights are hand-picked and the evaluation can only check coherence.
- Films with thin metadata still get lists that mostly share their genres (Casablanca gets romances rather than WWII films, Forrest Gump gets romantic comedies), and some franchises live only in titles: Saw's list has none of its six sequels.
- Well-known films are favoured on purpose. The lists reach about 62% of the catalogue, and films under 100 votes appear only when nothing better is similar.
- Non-English films rarely get recommendations in their own language: for the 298 non-English films, 2.9% of the picks share the film's original language.
- Filter search still scales popularity and vote count linearly, unlike similar movies.
- Lists need JavaScript, belong to one browser on one device and disappear when its site data is cleared.

## License and credits

Source available under the [PolyForm Noncommercial License 1.0.0](LICENSE): free for personal, educational and other noncommercial use; commercial use is not permitted.

Movie data and images come from [TMDB](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB. The base dataset is the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) on Kaggle.

The icons in `static/icons.svg` are from [Font Awesome Free 6.5.1](https://fontawesome.com) and stay under their own [CC BY 4.0 license](https://fontawesome.com/license/free). The type is [IBM Plex Sans and IBM Plex Sans Condensed](https://github.com/IBM/plex) (SIL Open Font License 1.1), loaded from Google Fonts.

## Türkçe özet

MovieRec, TMDB 5000 veri seti üzerinde çalışan, Flask ile yazılmış içerik tabanlı bir film öneri sistemidir. Beğendiğiniz bir filme benzeyen filmleri ya da seçtiğiniz tür, yönetmen, yıl ve puan filtrelerine uyan filmleri sıralar. Benzerlik için her filmin türleri, yönetmenleri, anahtar kelimeleri ve özet ile slogan metni ayrı TF-IDF bloklarına dönüştürülür ve her blok kendi ağırlığıyla birleştirilir; böylece özette geçen bir karakter adı ya da bir yönetmenin filmografisi sonucu tek başına belirleyemez. En benzer 91 film benzerlik, puan, popülerlik ve oy sayısıyla yeniden sıralanır; popülerlik ve oy sayısı logaritmik ölçeklenir. Ağırlıklar elle seçilmiştir. Kullanıcı verisi olmadığı için `scripts/evaluate.py` listeleri ölçer: tür örtüşmesi, aynı yönetmen, ortak başrol oyuncusu, seri filmlerini bulma oranı, katalog kapsamı ve önerilen filmlerin popülerliği. Bu ölçümler listelerin tutarlı olduğunu gösterir, insanların onları beğeneceğini göstermez.

Üyelik yoktur: beğeniler ve izleme listesi ziyaretçinin tarayıcısında saklanır, bu yüzden veritabanı gerekmez. Listeler sayfasındaki "Beğendiklerinize göre" bölümü, beğendiğiniz filmlerin benzer film listelerini birleştirir: birden çok beğeninizin listesinde üst sıralarda yer alan filmler öne çıkar, listelerinizdeki filmler gösterilmez. Proje ticari olmayan kullanım için PolyForm Noncommercial 1.0.0 lisansıyla paylaşılmıştır. Kurulum adımları yukarıdaki bölümlerdedir.
