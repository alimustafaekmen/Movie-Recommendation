import test from 'node:test';
import assert from 'node:assert/strict';

import { TOGGLES, iconHref, toggleView } from '../../static/js/movie_detail.js';

const [like, watchlist] = TOGGLES;
// The labels the page writes on each button (its dataset), in the page's language.
const LIKE_LABELS = { label: 'Beğen', activeLabel: 'Beğenildi' };
const WATCHLIST_LABELS = { label: 'Add to watchlist', activeLabel: 'On your watchlist' };

test('the like button names its state and swaps the outline heart for the solid one', () => {
  assert.equal(like.selector, '.btn-like');
  assert.deepEqual(toggleView(like, LIKE_LABELS, false), { label: 'Beğen', icon: 'heart-regular' });
  assert.deepEqual(toggleView(like, LIKE_LABELS, true), { label: 'Beğenildi', icon: 'heart' });
});

test('the watchlist button names its state and swaps the outline bookmark for the solid one', () => {
  assert.equal(watchlist.selector, '.btn-watchlist');
  assert.deepEqual(toggleView(watchlist, WATCHLIST_LABELS, false), { label: 'Add to watchlist', icon: 'bookmark-regular' });
  assert.deepEqual(toggleView(watchlist, WATCHLIST_LABELS, true), { label: 'On your watchlist', icon: 'bookmark' });
});

test('the icon swap keeps the versioned sprite URL and changes only the symbol', () => {
  assert.equal(iconHref('/static/icons.svg?v=0123456789ab#heart-regular', 'heart'), '/static/icons.svg?v=0123456789ab#heart');
  assert.equal(iconHref('/static/icons.svg?v=0123456789ab#bookmark', 'bookmark-regular'), '/static/icons.svg?v=0123456789ab#bookmark-regular');
  assert.equal(iconHref('/static/icons.svg', 'heart'), '/static/icons.svg#heart');
});
