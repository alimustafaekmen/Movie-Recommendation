import test from 'node:test';
import assert from 'node:assert/strict';

import { NOT_SAVED_MESSAGE } from '../../static/lists.js';
import { describeLists } from '../../static/my_lists.js';

const SOME_FAILED = 'Bazı filmler şu anda yüklenemedi.';
const card = (id, title) => ({ id, title, year: 2000, vote_average: 7, poster_url: null });
const CARDS = new Map([
  [1, card(1, 'Zodiac')],
  [2, card(2, 'Alien')],
  [3, card(3, 'Heat')],
]);
const titles = (sections) => sections.map((section) => section.cards.map((c) => c.title));

test('loaded lists show their cards in title order and no status', () => {
  const { sections, status } = describeLists([[1, 2], [3]], CARDS, true);

  assert.deepEqual(titles(sections), [['Alien', 'Zodiac'], ['Heat']]);
  assert.deepEqual(sections.map((section) => section.empty), [false, false]);
  assert.equal(status, '');
});

test('films that did not load are reported, not silently dropped', () => {
  const { sections, status } = describeLists([[1, 99], [3]], CARDS, true);

  assert.deepEqual(titles(sections), [['Zodiac'], ['Heat']]);
  assert.equal(status, SOME_FAILED);
});

test('a list whose films all failed to load is not shown as empty', () => {
  const { sections, status } = describeLists([[98, 99], []], CARDS, true);

  assert.deepEqual(sections, [
    { cards: [], empty: false },
    { cards: [], empty: true },
  ]);
  assert.equal(status, SOME_FAILED);
});

test('a store that cannot keep the lists shows the storage note', () => {
  const { sections, status } = describeLists([[], []], new Map(), false);

  assert.deepEqual(sections.map((section) => section.empty), [true, true]);
  assert.equal(status, NOT_SAVED_MESSAGE);
  assert.match(NOT_SAVED_MESSAGE, /kaydedemiyor/);
});
