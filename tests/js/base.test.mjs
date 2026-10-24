import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// static/js/base.js is a classic script at the end of <body>, so it runs here in a stand-in page
// whose elements are found by the exact selectors the script asks for.
const SOURCE = readFileSync(new URL('../../static/js/base.js', import.meta.url), 'utf8');
const settle = () => new Promise((resolve) => setImmediate(resolve));

class FakeElement {
  constructor(attributes = {}) {
    this.attributes = new Map(Object.entries(attributes));
    this.dataset = {};
    this.listeners = {};
    this.childNodes = [];
    this.hidden = false;
    this.textContent = '';
    this.tabIndex = 0;
    this.focused = false;
    this.queries = {};
  }

  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  removeAttribute(name) {
    this.attributes.delete(name);
  }

  addEventListener(type, fn) {
    (this.listeners[type] ??= []).push(fn);
  }

  removeEventListener(type, fn) {
    this.listeners[type] = (this.listeners[type] ?? []).filter((other) => other !== fn);
  }

  dispatch(type, event = {}) {
    let prevented = false;
    const full = { target: this, preventDefault: () => (prevented = true), ...event };
    [...(this.listeners[type] ?? [])].forEach((fn) => fn(full));
    return { prevented };
  }

  replaceChildren(...nodes) {
    this.childNodes = nodes;
  }

  querySelector(selector) {
    return (this.queries[selector] ?? [])[0] ?? null;
  }

  querySelectorAll(selector) {
    return this.queries[selector] ?? [];
  }

  focus() {
    this.focused = true;
  }
}

/** Run the script over a page: found maps the selectors of document.querySelectorAll to their
 * elements, byId the ids of document.getElementById; href is the page's URL. */
function loadPage({ found = {}, byId = {}, fetch = async () => ({ ok: false, status: 404 }), href = 'http://localhost/' } = {}) {
  const document = new FakeElement();
  document.queries = found;
  document.getElementById = (id) => byId[id] ?? null;
  const window = new FakeElement();
  window.location = { href };
  window.history = {
    state: { scroll: 1 },
    replaced: [],
    replaceState(state, unused, url) {
      this.replaced.push({ state, url });
    },
  };
  const timers = new Map();
  let nextTimer = 1;
  const context = vm.createContext({
    document,
    window,
    fetch,
    URL,
    Option: class {
      constructor(text) {
        this.text = text;
      }

      removeAttribute(name) {
        delete this[name];
      }
    },
    setTimeout: (fn, ms) => {
      timers.set(nextTimer, { fn, ms });
      return nextTimer++;
    },
    clearTimeout: (id) => timers.delete(id),
  });
  vm.runInContext(SOURCE, context);
  return { document, window, timers };
}

test('a page without the search panel, tabs or suggestions sets up without errors', () => {
  const { document, window } = loadPage();

  assert.doesNotThrow(() => {
    window.dispatch('pageshow', { persisted: true });
    document.dispatch('DOMContentLoaded');
  });
});

// --- Title and director suggestions ---
function suggestionPage(answers) {
  const list = new FakeElement();
  list.id = 'movie-titles';
  list.dataset.src = 'titles-url';
  const inputs = [new FakeElement(), new FakeElement()];
  const requests = [];
  const page = loadPage({
    found: { 'datalist[data-src]': [list], 'input[list="movie-titles"]': inputs },
    fetch: async (url) => {
      requests.push(url);
      return answers.shift()();
    },
  });
  return { ...page, list, inputs, requests };
}
const names = (list) => list.childNodes.map((option) => option.text);
const ok = (body) => () => ({ ok: true, status: 200, json: async () => body });

test('suggestions load on the first focus and only once', async () => {
  const page = suggestionPage([ok(['Heat', 'Alien'])]);

  page.inputs[0].dispatch('focus');
  await settle();
  page.inputs[1].dispatch('focus');
  page.inputs[0].dispatch('focus');
  await settle();

  assert.deepEqual(page.requests, ['titles-url']);
  assert.deepEqual(names(page.list), ['Heat', 'Alien']);
});

test('a failed suggestion fetch is tried again on the next focus', async () => {
  const offline = () => {
    throw new TypeError('Failed to fetch');
  };
  const page = suggestionPage([offline, ok(['Heat'])]);

  page.inputs[0].dispatch('focus');
  await settle();
  assert.deepEqual(names(page.list), []);

  page.inputs[1].dispatch('focus');
  await settle();

  assert.equal(page.requests.length, 2);
  assert.deepEqual(names(page.list), ['Heat']);
});

test('an error status or a body that is not a list of names is tried again', async () => {
  const unavailable = () => ({ ok: false, status: 503, json: async () => ({ status: 'error' }) });
  const notJson = () => ({ ok: true, status: 200, json: async () => JSON.parse('<!doctype html>') });
  const page = suggestionPage([
    unavailable,
    notJson,
    ok({ status: 'error' }),
    ok(['Heat', 7, null, [8, 'Sekiz'], ['Alien', 9]]),
  ]);

  for (let focus = 0; focus < 4; focus++) {
    page.inputs[0].dispatch('focus');
    await settle();
  }

  assert.equal(page.requests.length, 4);
  // Only names become options, never "null" or a number.
  assert.deepEqual(names(page.list), ['Heat', 'Alien']);
});

test('a focus while the suggestions load does not fetch them twice', async () => {
  let respond;
  const page = suggestionPage([() => new Promise((resolve) => (respond = resolve))]);

  page.inputs[0].dispatch('focus');
  page.inputs[1].dispatch('focus');
  respond(ok(['Heat'])());
  await settle();

  assert.equal(page.requests.length, 1);
  assert.deepEqual(names(page.list), ['Heat']);
});

test('an option shows its Turkish title while only that title holds the typed text', async () => {
  const page = suggestionPage([ok(['Heat', ['Inception', 'Başlangıç'], ['The Two Towers', 'İki Kule']])]);
  page.inputs[0].dispatch('focus');
  await settle();
  const labels = () => page.list.childNodes.map((option) => option.label ?? null);
  const type = (input, value) => {
    input.value = value;
    input.dispatch('input');
  };

  assert.deepEqual(names(page.list), ['Heat', 'Inception', 'The Two Towers']);
  assert.deepEqual(labels(), [null, null, null]);

  // Case and accents aside, as the server compares Turkish titles.
  type(page.inputs[0], 'BAŞLAN');
  assert.deepEqual(labels(), [null, 'Başlangıç', null]);
  type(page.inputs[1], 'iki');
  assert.deepEqual(labels(), [null, null, 'İki Kule']);
  // The title holds the text itself, so the option needs no second name.
  type(page.inputs[1], 'incep');
  assert.deepEqual(labels(), [null, null, null]);
  type(page.inputs[0], '');
  assert.deepEqual(labels(), [null, null, null]);
});

// --- The home page's search forms ---
function searchPage() {
  const label = { text: 'Benzerlerini bul' };
  const button = new FakeElement({ type: 'submit' });
  button.dataset.loadingText = 'Aranıyor...';
  button.childNodes = [label];
  const form = new FakeElement();
  form.queries = { '[data-loading-text]': [button] };
  const status = new FakeElement();
  const spinner = { spinner: true };
  const page = loadPage({
    found: { '.search-form': [form], '.search-status': [status] },
    byId: { 'spinner-icon-template': { content: { cloneNode: () => spinner } } },
  });
  return { ...page, form, button, status, label, spinner };
}

test('a search shows its loading text once and ignores a second submit', () => {
  const page = searchPage();

  assert.equal(page.form.dispatch('submit').prevented, false);

  assert.equal(page.button.getAttribute('aria-disabled'), 'true');
  assert.deepEqual(page.button.childNodes, [page.spinner, ' Aranıyor...']);
  assert.equal(page.status.textContent, 'Aranıyor...');
  assert.equal(page.form.dispatch('submit').prevented, true);
});

test('a page shown again from the back/forward cache gets its button back', () => {
  const page = searchPage();
  page.form.dispatch('submit');

  page.window.dispatch('pageshow', { persisted: true });

  assert.equal(page.button.getAttribute('aria-disabled'), null);
  assert.deepEqual(page.button.childNodes, [page.label]);
  assert.equal(page.status.textContent, '');
  assert.equal(page.timers.size, 0);
  assert.equal(page.form.dispatch('submit').prevented, false);
});

test('a search that never leaves the page (Stop pressed) frees the button after a while', () => {
  const page = searchPage();
  page.form.dispatch('submit');
  const [[, timer]] = page.timers;

  assert.ok(timer.ms >= 10000, String(timer.ms));
  timer.fn();

  assert.equal(page.button.getAttribute('aria-disabled'), null);
  assert.deepEqual(page.button.childNodes, [page.label]);
  assert.equal(page.form.dispatch('submit').prevented, false);
});

test('empty fields are left out of the URL unless nothing would be left', () => {
  const page = searchPage();
  const formData = new FormData();
  formData.append('film', 'Heat');
  formData.append('yonetmen', '');

  page.form.dispatch('formdata', { formData });
  assert.deepEqual([...formData.keys()], ['film']);

  const empty = new FormData();
  empty.append('film', '');
  page.form.dispatch('formdata', { formData: empty });
  assert.deepEqual([...empty.keys()], ['film']);
});

// --- Tabs ---
function tabsPage() {
  const tabs = ['similar', 'filter'].map((name, index) => {
    const tab = new FakeElement({ 'aria-controls': `${name}-panel`, 'aria-selected': String(index === 0) });
    return tab;
  });
  const panels = { 'similar-panel': new FakeElement(), 'filter-panel': new FakeElement() };
  const tablist = new FakeElement();
  tablist.hidden = true;
  tablist.queries = { '[role="tab"]': tabs };
  loadPage({ found: { '[role="tablist"]': [tablist] }, byId: panels });
  return { tablist, tabs, panels };
}

test('the tabs show the selected panel and move with the arrow keys', () => {
  const { tablist, tabs, panels } = tabsPage();
  assert.equal(tablist.hidden, false);
  assert.equal(panels['filter-panel'].hidden, true);

  const { prevented } = tablist.dispatch('keydown', { key: 'ArrowRight', target: tabs[0] });

  assert.equal(prevented, true);
  assert.equal(tabs[1].getAttribute('aria-selected'), 'true');
  assert.equal(tabs[1].focused, true);
  assert.equal(panels['similar-panel'].hidden, true);
});

test('arrow keys with Alt, Ctrl or Cmd are left to the browser, e.g. Back', () => {
  const { tablist, tabs } = tabsPage();

  for (const modifier of ['altKey', 'ctrlKey', 'metaKey']) {
    const { prevented } = tablist.dispatch('keydown', { key: 'ArrowLeft', target: tabs[0], [modifier]: true });

    assert.equal(prevented, false, modifier);
    assert.equal(tabs[0].getAttribute('aria-selected'), 'true', modifier);
  }
});

// --- Page modules (movie_detail.js, my_lists.js) that did not load ---
test('a status still marked when the page is parsed shows its load error', () => {
  const status = new FakeElement();
  status.dataset.loadError = 'Listeler şu anda kullanılamıyor.';
  const { document } = loadPage({ found: { '[data-load-error]': [status] } });
  // Module scripts run before DOMContentLoaded and remove the mark once they are set up.
  assert.equal(status.textContent, '');

  document.dispatch('DOMContentLoaded');

  assert.equal(status.textContent, 'Listeler şu anda kullanılamıyor.');
});

// --- The language switch ---
test('the ?lang= of the language switch leaves the address bar; the rest of the URL stays', () => {
  const { window } = loadPage({ href: 'http://localhost/?tur=Action&lang=en&tur=Drama#sonuc-13' });

  // The server stored the choice; an old ?lang= in the history must not store it again.
  assert.deepEqual(window.history.replaced, [
    { state: { scroll: 1 }, url: 'http://localhost/?tur=Action&tur=Drama#sonuc-13' },
  ]);
});

test('a page without ?lang= keeps its URL', () => {
  const { window } = loadPage({ href: 'http://localhost/movie/155?tab=cast' });

  assert.deepEqual(window.history.replaced, []);
});

test('a ?lang= alone leaves the bare path', () => {
  const { window } = loadPage({ href: 'http://localhost/my_lists?lang=tr' });

  assert.deepEqual(window.history.replaced.map(({ url }) => url), ['http://localhost/my_lists']);
});
