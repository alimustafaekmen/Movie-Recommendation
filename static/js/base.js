// Behaviour shared by every page: the language switch's ?lang=, flash messages, disclosure
// buttons, title suggestions for the header search, the home page's search panel, and a note
// when a page's module did not load.
// base.html loads this at the end of <body>, so the page is already parsed and the search panel
// is set up before it is first drawn.
(() => {
    // A results page takes a few seconds at most (the TMDb timeouts bound the poster requests).
    const LOADING_RESET_MS = 20000;

    // The header's language switch adds ?lang=, which the server has stored in a cookie by
    // now. It leaves the address bar, so Back, a reload or a bookmark of this URL cannot store
    // that language again over a later choice.
    const pageUrl = new URL(window.location.href);
    if (pageUrl.searchParams.has('lang')) {
        pageUrl.searchParams.delete('lang');
        window.history.replaceState(window.history.state, '', pageUrl.href);
    }

    // Delegated, so content added later (the my_lists.js toggles) works too.
    document.addEventListener('click', (event) => {
        const dismiss = event.target.closest('[data-dismiss="alert"]');
        if (dismiss) {
            dismiss.closest('.alert').remove();
        }

        // aria-expanded is the state. aria-controls, when given, names the element to hide;
        // without it the CSS reads aria-expanded (the home page's "Edit search").
        const toggle = event.target.closest('button[aria-expanded]');
        if (toggle) {
            const expanded = toggle.getAttribute('aria-expanded') === 'true';
            toggle.setAttribute('aria-expanded', String(!expanded));
            const target = document.getElementById(toggle.getAttribute('aria-controls'));
            if (target) {
                target.hidden = expanded;
            }
        }
    });

    // Text as the server compares Turkish titles (name_key in movierec/search_filters.py):
    // no case, no accents, punctuation as spaces. "Başlangıç" -> "baslangic"; \u0131 is the
    // dotless "ı", which has no accent to drop.
    const fold = (text) => text.normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase()
        .replaceAll('\u0131', 'i').replace(/[^\p{L}\p{N}]+/gu, ' ').trim();

    // The suggestion lists are large, so each is fetched when one of its inputs is first focused.
    // A failed fetch (offline, an error status, a body that is not a list) is tried again on the
    // next focus; the suggestions are optional, so typing works without them meanwhile.
    // A title whose film has a Turkish title comes as [title, Turkish title].
    document.querySelectorAll('datalist[data-src]').forEach((list) => {
        const inputs = document.querySelectorAll(`input[list="${list.id}"]`);
        let turkishTitles = [];
        const listen = (on) => inputs.forEach((input) => {
            if (on) {
                input.addEventListener('focus', load);
            } else {
                input.removeEventListener('focus', load);
            }
        });
        async function load() {
            listen(false);
            try {
                const response = await fetch(list.dataset.src);
                const names = response.ok ? await response.json() : null;
                if (!Array.isArray(names)) {
                    throw new TypeError('Expected a list of names');
                }
                const options = [];
                turkishTitles = [];
                names.forEach((name) => {
                    const [title, turkish] = Array.isArray(name) ? name : [name];
                    if (typeof title !== 'string') {
                        return;
                    }
                    const option = new Option(title);
                    options.push(option);
                    if (typeof turkish === 'string') {
                        turkishTitles.push({ option, turkish, titleKey: fold(title), turkishKey: fold(turkish) });
                    }
                });
                list.replaceChildren(...options);
            } catch {
                listen(true);
            }
        }
        listen(true);

        // The browser lists the options whose value or label holds the typed text. An option
        // found by its Turkish title shows that title as its label, so the visitor sees why it
        // is there; every other option shows its title alone.
        inputs.forEach((input) => input.addEventListener('input', () => {
            const typed = fold(input.value);
            turkishTitles.forEach(({ option, turkish, titleKey, turkishKey }) => {
                if (typed && turkishKey.includes(typed) && !titleKey.includes(typed)) {
                    option.label = turkish;
                } else {
                    option.removeAttribute('label');
                }
            });
        }));
    });

    // Tabs as in the WAI-ARIA tabs pattern: one tab stop, arrow keys, Home and End move
    // between tabs and selection follows focus.
    document.querySelectorAll('[role="tablist"]').forEach((tablist) => {
        const tabs = [...tablist.querySelectorAll('[role="tab"]')];
        const select = (chosen) => {
            tabs.forEach((tab) => {
                const selected = tab === chosen;
                tab.setAttribute('aria-selected', String(selected));
                tab.tabIndex = selected ? 0 : -1;
                document.getElementById(tab.getAttribute('aria-controls')).hidden = !selected;
            });
        };
        tablist.addEventListener('click', (event) => {
            const tab = event.target.closest('[role="tab"]');
            if (tab) {
                select(tab);
            }
        });
        tablist.addEventListener('keydown', (event) => {
            const current = tabs.indexOf(event.target);
            const moves = { ArrowLeft: current - 1, ArrowRight: current + 1, Home: 0, End: tabs.length - 1 };
            // With Alt, Ctrl or Cmd the keys are the browser's, e.g. Back.
            if (current < 0 || !(event.key in moves) || event.altKey || event.ctrlKey || event.metaKey) {
                return;
            }
            event.preventDefault();
            const tab = tabs[(moves[event.key] + tabs.length) % tabs.length];
            tab.focus();
            select(tab);
        });
        // The server shows every panel, so the forms also work without scripts; the tabs
        // take over from here, starting at the tab the server selected.
        select(tabs.find((tab) => tab.getAttribute('aria-selected') === 'true') ?? tabs[0]);
        tablist.hidden = false;
    });

    // On result pages the search panel folds behind "Edit search" so the results come first
    // on phones; home.css ignores the fold from 900px, where the panel has its own column.
    document.querySelectorAll('.search-toggle').forEach((toggle) => {
        toggle.setAttribute('aria-expanded', 'false');
        toggle.hidden = false;
    });

    // The home page searches are GET forms, so their URLs can be shared.
    const searchStatus = document.querySelector('.search-status');
    // The spinner's markup comes from the page, which knows the icon sprite's URL.
    const spinnerTemplate = document.getElementById('spinner-icon-template');
    document.querySelectorAll('.search-form').forEach((form) => {
        // Leave empty fields out of the URL, unless nothing would be left: the server then
        // still knows which search was asked for and explains what is missing.
        form.addEventListener('formdata', ({ formData }) => {
            const entries = [...formData];
            const empty = entries.filter(([, value]) => value === '');
            if (empty.length < entries.length) {
                empty.forEach(([name]) => formData.delete(name));
            }
        });
        // The results page can take a moment while posters load. The native required/min/max
        // checks run first, so this only shows for a search that is really sent. aria-disabled
        // (not disabled) keeps the focus on the button; a second submit is ignored instead.
        const button = form.querySelector('[data-loading-text]');
        let label = null;
        let resetTimer = 0;
        const reset = () => {
            clearTimeout(resetTimer);
            if (label) {
                button.replaceChildren(...label);
                label = null;
                button.removeAttribute('aria-disabled');
                searchStatus.textContent = '';
            }
        };
        form.addEventListener('submit', (event) => {
            if (label) {
                event.preventDefault();
                return;
            }
            label = [...button.childNodes];
            button.setAttribute('aria-disabled', 'true');
            button.replaceChildren(spinnerTemplate.content.cloneNode(true), ` ${button.dataset.loadingText}`);
            searchStatus.textContent = button.dataset.loadingText;
            // Stop (Esc) leaves the page as it is, and no event says so; well after the slowest
            // results page, the button takes searches again.
            resetTimer = setTimeout(reset, LOADING_RESET_MS);
        });
        // Back/forward can show this page again from the browser's cache, still "loading".
        window.addEventListener('pageshow', reset);
    });

    // movie_detail.js and my_lists.js remove data-load-error from their status line once they
    // are set up. Module scripts run before DOMContentLoaded, so a mark left then means the
    // module did not load (e.g. a network error on it or on list_store.js) and its buttons do
    // nothing.
    document.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('[data-load-error]').forEach((status) => {
            status.textContent = status.dataset.loadError;
        });
    });
})();
