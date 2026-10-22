import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// static/js/theme.js is a classic script (it must block in <head>), so it runs here in a stand-in
// page instead of being imported.
const SOURCE = readFileSync(new URL('../../static/js/theme.js', import.meta.url), 'utf8');
const KEY = 'movierec:theme';

function makeButton() {
  return {
    attributes: {}, title: '', onClick: [],
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(_, fn) { this.onClick.push(fn); },
  };
}

/** Run the script with a stored choice (null for none), the system's preference, and storage
 * that can throw like a blocked localStorage. The buttons exist once ready() parses the body.
 * media: 'none' for a browser without matchMedia, 'old' for one whose MediaQueryList has no
 * addEventListener (Safari before 14). */
function loadPage({ stored = null, systemDark = false, blockedStorage = false, media: kind = 'modern', buttons = 1 } = {}) {
  const storage = new Map(stored === null ? [] : [[KEY, stored]]);
  const media = { matches: systemDark, onChange: [] };
  if (kind === 'modern') media.addEventListener = (_, fn) => media.onChange.push(fn);
  const toggles = Array.from({ length: buttons }, makeButton);
  const onDocument = {};
  const onWindow = {};
  let parsed = false;
  const document = {
    documentElement: { dataset: {} },
    querySelector: (selector) => (parsed && selector === '[data-theme-toggle]' ? toggles[0] ?? null : null),
    querySelectorAll: (selector) => (parsed && selector === '[data-theme-toggle]' ? toggles : []),
    addEventListener: (type, fn) => { onDocument[type] = fn; },
  };
  const guard = () => { if (blockedStorage) throw new Error('storage blocked'); };
  const localStorage = {
    getItem: (key) => { guard(); return storage.has(key) ? storage.get(key) : null; },
    setItem: (key, value) => { guard(); storage.set(key, String(value)); },
  };
  const window = {
    addEventListener: (type, fn) => { (onWindow[type] ??= []).push(fn); },
    ...(kind === 'none' ? {} : { matchMedia: () => media }),
  };
  const context = vm.createContext({ document, localStorage, window });
  vm.runInContext(SOURCE, context);
  const fire = (type, event) => (onWindow[type] ?? []).forEach((fn) => fn(event));
  return {
    context, storage, button: toggles[0], toggles,
    theme: () => document.documentElement.dataset.theme,
    ready() { parsed = true; onDocument.DOMContentLoaded(); },
    click(index = 0) { toggles[index].onClick.forEach((fn) => fn()); },
    systemTurns(dark) { media.matches = dark; media.onChange.forEach((fn) => fn()); },
    // Another tab stored a choice (null: cleared the storage).
    otherTabStores(value) {
      if (value === null) storage.clear(); else storage.set(KEY, value);
      fire('storage', { key: value === null ? null : KEY });
    },
    // This page comes back from the back/forward cache after the choice changed elsewhere.
    restoredWith(value) { storage.set(KEY, value); fire('pageshow', { persisted: true }); },
  };
}

test('the stored choice wins over the system, which decides otherwise', () => {
  const { effectiveTheme } = loadPage().context;

  assert.equal(effectiveTheme('dark', false), 'dark');
  assert.equal(effectiveTheme('light', true), 'light');
  assert.equal(effectiveTheme(null, true), 'dark');
  assert.equal(effectiveTheme(null, false), 'light');
  // Anything else in storage counts as no choice.
  assert.equal(effectiveTheme('sepia', true), 'dark');
});

test('the button switches to the other theme', () => {
  const { nextTheme } = loadPage().context;

  assert.equal(nextTheme('light'), 'dark');
  assert.equal(nextTheme('dark'), 'light');
});

test('a stored dark choice is on <html> before the body is parsed', () => {
  assert.equal(loadPage({ stored: 'dark' }).theme(), 'dark');
  assert.equal(loadPage({ stored: 'light', systemDark: true }).theme(), 'light');
});

test('without a choice the page follows the system, also when it changes', () => {
  const page = loadPage({ systemDark: true });
  assert.equal(page.theme(), 'dark');

  page.systemTurns(false);

  assert.equal(page.theme(), 'light');
});

test('a stored choice ignores a change of the system', () => {
  const page = loadPage({ stored: 'light' });

  page.systemTurns(true);

  assert.equal(page.theme(), 'light');
});

test('the button names the theme it switches to', () => {
  const dark = loadPage({ stored: 'dark' });
  dark.ready();
  assert.equal(dark.button.attributes['aria-label'], 'Açık temaya geç');
  assert.equal(dark.button.title, 'Açık temaya geç');

  const light = loadPage();
  light.ready();
  assert.equal(light.button.attributes['aria-label'], 'Koyu temaya geç');
  assert.equal(light.button.title, 'Koyu temaya geç');
});

test('a click flips the drawn theme, stores it and relabels the button', () => {
  const page = loadPage({ systemDark: true });
  page.ready();

  page.click();

  assert.equal(page.theme(), 'light');
  assert.equal(page.storage.get(KEY), 'light');
  assert.equal(page.button.attributes['aria-label'], 'Koyu temaya geç');

  page.click();

  assert.equal(page.theme(), 'dark');
  assert.equal(page.storage.get(KEY), 'dark');
  assert.equal(page.button.attributes['aria-label'], 'Açık temaya geç');
});

test('blocked storage leaves the theme to the system and the button still switches', () => {
  const page = loadPage({ systemDark: true, blockedStorage: true });
  assert.equal(page.theme(), 'dark');
  page.ready();

  page.click();

  assert.equal(page.theme(), 'light');
});

test('without matchMedia the page is light unless a choice is stored, and the button works', () => {
  const page = loadPage({ media: 'none' });
  assert.equal(page.theme(), 'light');
  assert.equal(loadPage({ media: 'none', stored: 'dark' }).theme(), 'dark');
  page.ready();

  page.click();

  assert.equal(page.theme(), 'dark');
  assert.equal(page.storage.get(KEY), 'dark');
});

test('a MediaQueryList without addEventListener (Safari 13) still leaves a working button', () => {
  const page = loadPage({ media: 'old', systemDark: true });
  assert.equal(page.theme(), 'dark');
  page.ready();

  page.click();

  assert.equal(page.theme(), 'light');
  assert.equal(page.button.attributes['aria-label'], 'Koyu temaya geç');
});

test('every theme button is labelled and switches the theme', () => {
  const page = loadPage({ buttons: 2 });
  page.ready();
  assert.deepEqual(page.toggles.map((button) => button.attributes['aria-label']), ['Koyu temaya geç', 'Koyu temaya geç']);

  page.click(1);

  assert.equal(page.theme(), 'dark');
  assert.deepEqual(page.toggles.map((button) => button.title), ['Açık temaya geç', 'Açık temaya geç']);
});

test('a choice made in another tab is drawn here too', () => {
  const page = loadPage();
  page.ready();

  page.otherTabStores('dark');
  assert.equal(page.theme(), 'dark');
  assert.equal(page.button.attributes['aria-label'], 'Açık temaya geç');

  page.otherTabStores(null);
  assert.equal(page.theme(), 'light');
});

test('a page restored from the back/forward cache draws the current choice', () => {
  const page = loadPage({ stored: 'dark' });
  page.ready();

  page.restoredWith('light');

  assert.equal(page.theme(), 'light');
  assert.equal(page.button.attributes['aria-label'], 'Koyu temaya geç');
});
