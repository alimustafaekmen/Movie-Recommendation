// Behaviour shared by every page. Loaded at the end of <body>, so the page is already parsed.
(() => {
    // Well past the slowest results page; TMDb timeouts bound the poster requests.
    const LOADING_RESET_MS = 20000;

    // The cookie now holds ?lang=; dropping it keeps a reload or bookmark from resetting it.
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

        // aria-controls names the element to hide; without it the CSS reads aria-expanded.
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

    // Mirrors name_key in movierec/search_filters.py: "Başlangıç" -> "baslangic".
    const fold = (text) => text.normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase()
        .replaceAll('\u0131', 'i').replace(/[^\p{L}\p{N}]+/gu, ' ').trim();

    // The lists are large, so each loads on first focus and retries on the next if it fails.
    // Entries are a title or [title, Turkish title].
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

        // An option matched by its Turkish title shows that title, so the visitor sees why.
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

    // WAI-ARIA tabs pattern: one tab stop, arrows, Home and End; selection follows focus.
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
        // The server shows every panel, so the forms work without scripts.
        select(tabs.find((tab) => tab.getAttribute('aria-selected') === 'true') ?? tabs[0]);
        tablist.hidden = false;
    });

    // On phones the results come first; home.css ignores the fold from 900px.
    document.querySelectorAll('.search-toggle').forEach((toggle) => {
        toggle.setAttribute('aria-expanded', 'false');
        toggle.hidden = false;
    });

    const searchStatus = document.querySelector('.search-status');
    // From the page, which knows the icon sprite's URL.
    const spinnerTemplate = document.getElementById('spinner-icon-template');
    document.querySelectorAll('.search-form').forEach((form) => {
        // Drop empty fields, unless all are empty, so the server can say what is missing.
        form.addEventListener('formdata', ({ formData }) => {
            const entries = [...formData];
            const empty = entries.filter(([, value]) => value === '');
            if (empty.length < entries.length) {
                empty.forEach(([name]) => formData.delete(name));
            }
        });
        // aria-disabled, not disabled, so the button keeps focus; a second submit is ignored.
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
            // Stopping the load fires no event, so the button resets itself.
            resetTimer = setTimeout(reset, LOADING_RESET_MS);
        });
        // Back/forward can restore this page from cache, still "loading".
        window.addEventListener('pageshow', reset);
    });

    // Modules clear data-load-error once set up and run before DOMContentLoaded, so a mark
    // left here means a module failed to load.
    document.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('[data-load-error]').forEach((status) => {
            status.textContent = status.dataset.loadError;
        });
    });
})();
