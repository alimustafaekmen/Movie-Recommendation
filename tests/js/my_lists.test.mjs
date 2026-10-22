import test from 'node:test';
import assert from 'node:assert/strict';

import { NOT_SAVED_MESSAGE } from '../../static/js/list_store.js';
import {
  cardContent,
  describeLists,
  fetchCards,
  fetchRecommendations,
  formatRating,
  movieHref,
  recommendationsToShow,
} from '../../static/js/my_lists.js';

const SOME_FAILED = 'Bazı filmler şu anda yüklenemedi.';
const card = (id, title) => ({ id, title, year: 2000, vote_average: 7, poster_url: null });
const CARDS = new Map([
  [1, card(1, 'Zodiac')],
  [2, card(2, 'Alien')],
  [3, card(3, 'Heat')],
]);
const titles = (sections) => sections.map((cards) => cards.map((c) => c.title));
// The page's data-movie-url: the server's URL of the movie page for id 0.
const MOVIE_URL = '/movie/0';

test('loaded lists show their cards in title order and no status', () => {
  const { sections, status } = describeLists([[1, 2], [3]], CARDS, true);

  assert.deepEqual(titles(sections), [['Alien', 'Zodiac'], ['Heat']]);
  assert.equal(status, '');
});

test('films that did not load are reported, not silently dropped', () => {
  const { sections, status } = describeLists([[1, 99], [3]], CARDS, true);

  assert.deepEqual(titles(sections), [['Zodiac'], ['Heat']]);
  assert.equal(status, SOME_FAILED);
});

test('a list whose films all failed to load is reported', () => {
  const { sections, status } = describeLists([[98, 99], []], CARDS, true);

  assert.deepEqual(sections, [[], []]);
  assert.equal(status, SOME_FAILED);
});

test('a store that cannot keep the lists shows the storage note', () => {
  const { sections, status } = describeLists([[], []], new Map(), false);

  assert.deepEqual(sections, [[], []]);
  assert.equal(status, NOT_SAVED_MESSAGE);
  assert.match(NOT_SAVED_MESSAGE, /kaydedemiyor/);
});

test('ratings use the decimal comma and round like the format_rating filter', () => {
  const cases = [
    [8.2, '8,2'],
    [7, '7,0'],
    [10, '10,0'],
    [6.66, '6,7'],
    [8.25, '8,3'],
    [8.35, '8,4'],
    [null, ''],
    [0, ''],
    // Not a number from the API: nothing rather than an unformatted or odd value.
    ['8.2', ''],
    [NaN, ''],
    [Infinity, ''],
    [undefined, ''],
  ];
  for (const [value, expected] of cases) {
    assert.equal(formatRating(value), expected, String(value));
  }
});

test('a movie link is the server URL for id 0 with the id in place of the 0', () => {
  assert.equal(movieHref(MOVIE_URL, 155), '/movie/155');
  // Only the id changes, wherever the app is mounted and however the route ends.
  assert.equal(movieHref('/v10/movie/0', 7), '/v10/movie/7');
  assert.equal(movieHref('/movie/0/', 2147483647), '/movie/2147483647/');
});

test('a card links to its movie and shows its year, rating and poster', () => {
  const poster = 'https://image.tmdb.org/t/p/w342/dk.jpg';
  const shown = cardContent({ id: 155, title: 'The Dark Knight', year: 2008, vote_average: 8.2, poster_url: poster }, MOVIE_URL);

  assert.deepEqual(shown, {
    href: '/movie/155',
    title: 'The Dark Knight',
    year: '2008',
    rating: '8,2',
    poster,
  });
});

test('a card leaves out an unknown year, a missing rating and a missing poster', () => {
  const shown = cardContent({ id: 7, title: 'Heat', year: 0, vote_average: 0, poster_url: null }, MOVIE_URL);

  assert.deepEqual(shown, { href: '/movie/7', title: 'Heat', year: '', rating: '', poster: '' });
});

test('a card with wrongly typed fields shows only what it can', () => {
  const shown = cardContent({ id: 7, title: 'Heat', year: '1995', vote_average: '8', poster_url: { url: 'x' } }, MOVIE_URL);

  assert.deepEqual(shown, { href: '/movie/7', title: 'Heat', year: '', rating: '', poster: '' });
});

// --- /api/movies, with fetch replaced by the answers each request gets ---
const CARDS_URL = '/api/movies';
const json = (body, status = 200) => ({ ok: status === 200, status, json: async () => body });

async function withFetch(answer, run) {
  const previous = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url) => {
    requests.push(url);
    return answer(new URL(url, 'http://localhost').searchParams.get('ids').split(',').map(Number));
  };
  try {
    return { result: await run(), requests };
  } finally {
    globalThis.fetch = previous;
  }
}

test('cards are fetched in chunks of 20 ids, as many as the server looks up on TMDb', async () => {
  const ids = Array.from({ length: 250 }, (_, index) => index + 1);
  const { result, requests } = await withFetch((asked) => json(asked.map((id) => card(id, `Film ${id}`))), () => fetchCards(CARDS_URL, ids));

  assert.equal(requests.length, 13);
  assert.equal(requests[0], `${CARDS_URL}?ids=${ids.slice(0, 20).join(',')}`);
  assert.equal(result.cards.size, 250);
  assert.equal(result.failed, false);
});

test('a chunk that fails keeps the cards of the others and is reported', async () => {
  const ids = Array.from({ length: 30 }, (_, index) => index + 1);
  const answers = [
    (asked) => json(asked.map((id) => card(id, `Film ${id}`))),
    () => {
      throw new TypeError('Failed to fetch');
    },
  ];
  const { result } = await withFetch((asked) => answers.shift()(asked), () => fetchCards(CARDS_URL, ids));

  assert.equal(result.cards.size, 20);
  assert.equal(result.failed, true);
});

test('error statuses, bodies that are not JSON and JSON that is not a list count as failed', async () => {
  const answers = [
    () => json({ status: 'error', error: 'unavailable' }, 503),
    () => ({ ok: true, status: 200, json: async () => JSON.parse('<!doctype html>') }),
    () => json({ status: 'error' }),
    () => json(null),
  ];
  for (const answer of answers) {
    const { result } = await withFetch(answer, () => fetchCards(CARDS_URL, [1, 2]));

    assert.equal(result.cards.size, 0);
    assert.equal(result.failed, true);
  }
});

test('malformed cards are left out, so the page reports them as not loaded', async () => {
  const { result } = await withFetch(
    () => json([null, 7, 'x', { id: '1', title: 'A' }, { id: 2 }, { id: 3, title: 42 }, card(4, 'Heat')]),
    () => fetchCards(CARDS_URL, [1, 2, 3, 4]),
  );

  assert.deepEqual([...result.cards.keys()], [4]);
  assert.equal(result.failed, false);
  assert.equal(describeLists([[1, 2, 3, 4]], result.cards, true).status, SOME_FAILED);
});

// --- /api/recommendations ---
const RECOMMENDATIONS_URL = '/api/recommendations';

async function withAnswer(answer, run) {
  const previous = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url) => {
    requests.push(url);
    return answer();
  };
  try {
    return { result: await run(), requests };
  } finally {
    globalThis.fetch = previous;
  }
}

test('without favourites nothing is asked for and the section stays hidden', async () => {
  const { result, requests } = await withAnswer(() => json([card(1, 'Heat')]), () => fetchRecommendations(RECOMMENDATIONS_URL, [], [5]));

  assert.deepEqual(result, []);
  assert.deepEqual(requests, []);
  assert.deepEqual(recommendationsToShow([], [5], [card(1, 'Heat')]), []);
});

test('recommendations are asked for with the first 100 ids of each list and shown in order', async () => {
  const favorites = Array.from({ length: 150 }, (_, index) => index + 1000);
  const answer = [card(7, 'Zodiac'), null, { id: 8 }, card(9, 'Alien')];
  const { result, requests } = await withAnswer(() => json(answer), () => fetchRecommendations(RECOMMENDATIONS_URL, favorites, [200, 201]));

  assert.deepEqual(requests, [`${RECOMMENDATIONS_URL}?favorites=${favorites.slice(0, 100).join(',')}&watchlist=200,201`]);
  // Malformed cards are left out; the server's order is kept, not sorted by title.
  assert.deepEqual(recommendationsToShow(favorites, [200, 201], result).map((c) => c.title), ['Zodiac', 'Alien']);
});

test('films already on a list are never recommended', () => {
  // The server reads only the first 100 ids of a list, so the page leaves out the others.
  const cards = [card(1, 'Heat'), card(2, 'Alien'), card(3, 'Zodiac')];

  assert.deepEqual(recommendationsToShow([2], [3], cards).map((c) => c.id), [1]);
});

test('a failed recommendations request shows nothing', async () => {
  const answers = [
    () => json({ status: 'error', error: 'unavailable' }, 503),
    () => json({ status: 'error' }),
    () => {
      throw new TypeError('Failed to fetch');
    },
  ];
  for (const answer of answers) {
    const { result } = await withAnswer(answer, () => fetchRecommendations(RECOMMENDATIONS_URL, [1], []));

    assert.deepEqual(result, []);
  }
});
