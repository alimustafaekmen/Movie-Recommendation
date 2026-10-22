import test from 'node:test';
import assert from 'node:assert/strict';

import { createListStore, openListStore, watchLists } from '../../static/js/list_store.js';

// A Storage stand-in; the methods named in failOn throw like a full or blocked localStorage.
function memoryStorage(initial = {}, { failOn = [] } = {}) {
  const items = new Map(Object.entries(initial));
  const guard = (method) => {
    if (failOn.includes(method)) throw new Error(`${method} blocked`);
  };
  return {
    getItem(key) {
      guard('getItem');
      return items.has(key) ? items.get(key) : null;
    },
    setItem(key, value) {
      guard('setItem');
      items.set(key, String(value));
    },
    removeItem(key) {
      guard('removeItem');
      items.delete(key);
    },
    keys: () => [...items.keys()],
  };
}

test('toggle, has and list work per list', () => {
  const store = createListStore(memoryStorage());

  assert.equal(store.has('favorites', 155), false);
  assert.equal(store.toggle('favorites', 155), true);
  assert.equal(store.has('favorites', 155), true);
  assert.equal(store.has('watchlist', 155), false);
  assert.equal(store.toggle('watchlist', 19995), true);
  assert.deepEqual(store.list('favorites'), [155]);
  assert.deepEqual(store.list('watchlist'), [19995]);

  assert.equal(store.toggle('favorites', 155), false);
  assert.equal(store.has('favorites', 155), false);
  assert.deepEqual(store.list('favorites'), []);
  assert.deepEqual(store.list('watchlist'), [19995]);
});

test('list returns a copy, newest first', () => {
  const store = createListStore(memoryStorage());
  store.toggle('watchlist', 1);
  store.toggle('watchlist', 2);

  const ids = store.list('watchlist');
  ids.push(3);

  assert.deepEqual(store.list('watchlist'), [2, 1]);
});

test('lists are stored as JSON id arrays under the movierec keys', () => {
  const storage = memoryStorage();
  const store = createListStore(storage);
  store.toggle('favorites', 155);
  store.toggle('favorites', 19995);
  store.toggle('watchlist', 27205);

  assert.equal(storage.getItem('movierec:favorites'), '[19995,155]');
  assert.equal(storage.getItem('movierec:watchlist'), '[27205]');
  assert.deepEqual(storage.keys().sort(), ['movierec:favorites', 'movierec:watchlist']);
  assert.equal(store.persistent, true);

  const reopened = createListStore(storage);
  assert.deepEqual(reopened.list('favorites'), [19995, 155]);
  assert.equal(reopened.has('watchlist', 27205), true);
});

test('changes made by another tab are seen on the next read', () => {
  const storage = memoryStorage();
  const tabA = createListStore(storage);
  const tabB = createListStore(storage);

  tabA.toggle('favorites', 1);
  tabB.toggle('favorites', 2);

  assert.deepEqual(tabA.list('favorites'), [2, 1]);
});

test('corrupt or unexpected stored values read as an empty list', () => {
  for (const raw of ['{not json', '', '"155"', '{"a":1}', 'null', '155']) {
    const storage = memoryStorage({ 'movierec:favorites': raw });
    const store = createListStore(storage);

    assert.deepEqual(store.list('favorites'), [], raw);
    assert.equal(store.toggle('favorites', 7), true, raw);
    assert.equal(storage.getItem('movierec:favorites'), '[7]', raw);
  }
});

test('invalid and duplicate stored ids are dropped', () => {
  const raw = '[155, "x", "19995", -1, 0, 2.5, 155, 2147483648, null, 27205]';
  const store = createListStore(memoryStorage({ 'movierec:watchlist': raw }));

  assert.deepEqual(store.list('watchlist'), [155, 27205]);
});

test('a storage that throws keeps the lists in memory and is not persistent', () => {
  const store = createListStore(memoryStorage({}, { failOn: ['getItem', 'setItem'] }));

  assert.equal(store.persistent, false);
  assert.equal(store.toggle('favorites', 155), true);
  assert.equal(store.has('favorites', 155), true);
  assert.deepEqual(store.list('favorites'), [155]);
});

test('a storage that cannot write keeps both lists in memory', () => {
  const storage = memoryStorage({ 'movierec:favorites': '[1]' }, { failOn: ['setItem'] });
  const store = createListStore(storage);

  assert.equal(store.toggle('watchlist', 2), true);

  assert.equal(store.persistent, false);
  assert.deepEqual(store.list('watchlist'), [2]);
  assert.deepEqual(store.list('favorites'), [1]);
});

test('missing storage works in memory and is not persistent', () => {
  for (const storage of [undefined, null]) {
    const store = createListStore(storage);

    assert.equal(store.persistent, false);
    assert.equal(store.toggle('watchlist', 9), true);
    assert.deepEqual(store.list('watchlist'), [9]);
  }
});

test('openListStore falls back to memory when localStorage cannot be reached', () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'window');
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {
      get localStorage() {
        throw new Error('SecurityError');
      },
    },
  });
  try {
    const store = openListStore();
    assert.equal(store.persistent, false);
    assert.equal(store.toggle('favorites', 1), true);
  } finally {
    if (previous) Object.defineProperty(globalThis, 'window', previous);
    else delete globalThis.window;
  }
});

test('invalid ids and unknown lists are rejected', () => {
  const store = createListStore(memoryStorage());

  for (const id of [0, -1, 1.5, '155', NaN, null, undefined, 2147483648]) {
    assert.throws(() => store.toggle('favorites', id), TypeError, String(id));
    assert.equal(store.has('favorites', id), false, String(id));
  }
  assert.throws(() => store.toggle('likes', 1), TypeError);
  assert.throws(() => store.list('likes'), TypeError);
  assert.throws(() => store.has('likes', 1), TypeError);
  assert.deepEqual(store.list('favorites'), []);
});

// --- Lists saved by earlier versions of the site under the old "film-oneri:" keys ---
test('lists under the old keys move to the new keys', () => {
  const storage = memoryStorage({ 'film-oneri:favorites': '[3,1,2]', 'film-oneri:watchlist': '[9]' });
  const store = createListStore(storage);

  assert.deepEqual(store.list('favorites'), [3, 1, 2]);
  assert.deepEqual(store.list('watchlist'), [9]);
  assert.equal(store.persistent, true);
  assert.equal(storage.getItem('movierec:favorites'), '[3,1,2]');
  assert.equal(storage.getItem('movierec:watchlist'), '[9]');
  assert.deepEqual(storage.keys().sort(), ['movierec:favorites', 'movierec:watchlist']);
});

test('an old list joins a new one after its own ids, in order and without duplicates', () => {
  // E.g. a tab still running the old script kept writing the old key after the move.
  const storage = memoryStorage({ 'movierec:favorites': '[5,1]', 'film-oneri:favorites': '[1,7,5,8,7]' });
  const store = createListStore(storage);

  assert.deepEqual(store.list('favorites'), [5, 1, 7, 8]);
  assert.equal(storage.getItem('movierec:favorites'), '[5,1,7,8]');
  assert.equal(storage.getItem('film-oneri:favorites'), null);
  assert.deepEqual(store.list('watchlist'), []);
});

test('a corrupt old list is dropped and the new one is kept', () => {
  const storage = memoryStorage({ 'movierec:watchlist': '[4]', 'film-oneri:watchlist': '{not json' });
  const store = createListStore(storage);

  assert.deepEqual(store.list('watchlist'), [4]);
  assert.equal(storage.getItem('film-oneri:watchlist'), null);
});

test('an old list that cannot be saved under the new key stays where it was', () => {
  const storage = memoryStorage({ 'film-oneri:favorites': '[1,2]' }, { failOn: ['setItem'] });
  const store = createListStore(storage);

  // This page still shows it; the next visit tries the move again.
  assert.deepEqual(store.list('favorites'), [1, 2]);
  assert.equal(store.persistent, false);
  assert.equal(storage.getItem('film-oneri:favorites'), '[1,2]');
  assert.equal(storage.getItem('movierec:favorites'), null);
});

test('an old key that cannot be removed is merged again without duplicates', () => {
  const storage = memoryStorage({ 'film-oneri:favorites': '[1,2]' }, { failOn: ['removeItem'] });

  const first = createListStore(storage);
  first.toggle('favorites', 3);
  const second = createListStore(storage);

  assert.equal(first.persistent, true);
  assert.deepEqual(second.list('favorites'), [3, 1, 2]);
  assert.equal(storage.getItem('movierec:favorites'), '[3,1,2]');
});

test('a storage that cannot be read moves nothing and keeps the lists in memory', () => {
  const storage = memoryStorage({ 'film-oneri:favorites': '[1]' }, { failOn: ['getItem'] });
  const store = createListStore(storage);

  assert.equal(store.persistent, false);
  assert.deepEqual(store.list('favorites'), []);
  assert.equal(store.toggle('favorites', 2), true);
  assert.deepEqual(store.list('favorites'), [2]);
});

// --- Changes made outside this page ---
const storageEvent = (key) => Object.assign(new Event('storage'), { key });
const pageshowEvent = (persisted) => Object.assign(new Event('pageshow'), { persisted });

test('a list change in another tab calls back; other keys do not', () => {
  const target = new EventTarget();
  let calls = 0;
  watchLists(() => calls++, target);

  target.dispatchEvent(storageEvent('movierec:favorites'));
  target.dispatchEvent(storageEvent('movierec:watchlist'));
  assert.equal(calls, 2);

  target.dispatchEvent(storageEvent('movierec:theme'));
  target.dispatchEvent(storageEvent('film-oneri:favorites'));
  assert.equal(calls, 2);

  // Another tab cleared the storage.
  target.dispatchEvent(storageEvent(null));
  assert.equal(calls, 3);
});

test('a page shown again from the back/forward cache calls back; a fresh load does not', () => {
  const target = new EventTarget();
  let calls = 0;
  watchLists(() => calls++, target);

  target.dispatchEvent(pageshowEvent(false));
  assert.equal(calls, 0);

  target.dispatchEvent(pageshowEvent(true));
  assert.equal(calls, 1);
});
