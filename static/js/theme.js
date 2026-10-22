// The light or dark theme. base.html loads this classic script in <head>, before the
// stylesheets and blocking on purpose, so <html> carries data-theme before the first paint and
// a stored dark choice never flashes light. css/base.css draws the dark tokens for
// data-theme="dark". The header button flips the theme and this browser remembers the choice.

/** The theme to draw: the visitor's stored choice, else the system's preference. */
function effectiveTheme(stored, systemDark) {
    if (stored === 'light' || stored === 'dark') {
        return stored;
    }
    return systemDark ? 'dark' : 'light';
}

/** The theme a click on the button switches to. */
function nextTheme(current) {
    return current === 'dark' ? 'light' : 'dark';
}

(() => {
    const KEY = 'movierec:theme';
    // The label says what a click does, so it names the other theme.
    const LABELS = { light: 'Koyu temaya geç', dark: 'Açık temaya geç' };
    const root = document.documentElement;
    // Missing in very old browsers, which then draw the light theme unless a choice is stored.
    const darkSchemeQuery = window.matchMedia?.('(prefers-color-scheme: dark)');

    // Blocked storage (private windows, disabled site data) leaves the theme to the system.
    function storedTheme() {
        try {
            return localStorage.getItem(KEY);
        } catch {
            return null;
        }
    }

    function apply(theme) {
        root.dataset.theme = theme;
        document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
            button.setAttribute('aria-label', LABELS[theme]);
            button.title = LABELS[theme];
        });
    }

    const sync = () => apply(effectiveTheme(storedTheme(), Boolean(darkSchemeQuery?.matches)));
    sync();

    // Without a stored choice the page keeps following the system, also when it changes.
    // Safari before 14 lacks addEventListener here; such pages follow the system on load only.
    darkSchemeQuery?.addEventListener?.('change', sync);
    // The choice may change in another tab, or on another page while this one waits in the
    // back/forward cache.
    window.addEventListener('storage', (event) => {
        if (event.key === KEY || event.key === null) {
            sync();
        }
    });
    window.addEventListener('pageshow', (event) => {
        if (event.persisted) {
            sync();
        }
    });

    document.addEventListener('DOMContentLoaded', () => {
        apply(root.dataset.theme);
        document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
            button.addEventListener('click', () => {
                const theme = nextTheme(root.dataset.theme);
                try {
                    localStorage.setItem(KEY, theme);
                } catch {
                    // Not remembered, but this page still switches.
                }
                apply(theme);
            });
        });
    });
})();
