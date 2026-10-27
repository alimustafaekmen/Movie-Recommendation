// The movie detail page: like and watchlist toggles, and the trailer that loads on click.
// The lists live in this browser, so the server renders both buttons inactive and this module
// sets their real state.
import { openListStore, watchLists } from './list_store.js';

// The label alone says the state, so screen readers and sighted visitors get the same signal;
// the page writes both labels on each button (data-label, data-active-label) in its language.
// The icon follows: the outline symbol of static/icons.svg while off, the solid one while on.
export const TOGGLES = [
    { selector: '.btn-like', list: 'favorites', activeClass: 'liked', icon: 'heart-regular', activeIcon: 'heart' },
    { selector: '.btn-watchlist', list: 'watchlist', activeClass: 'added', icon: 'bookmark-regular', activeIcon: 'bookmark' },
];
const PLAYER_PERMISSIONS = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';

/**
 * The label and the sprite icon a toggle shows while on (active) or off; labels is the button's
 * dataset, with label and activeLabel.
 */
export function toggleView(toggle, labels, active) {
    return active
        ? { label: labels.activeLabel, icon: toggle.activeIcon }
        : { label: labels.label, icon: toggle.icon };
}

/** A sprite URL such as '/static/icons.svg?v=…#heart-regular' pointed at another symbol. */
export function iconHref(href, name) {
    return `${href.replace(/#.*$/, '')}#${name}`;
}

// Both shapes of an icon share a viewBox, so swapping the symbol is all it takes.
function setState(button, toggle, active) {
    const { label, icon } = toggleView(toggle, button.dataset, active);
    button.classList.toggle(toggle.activeClass, active);
    button.querySelector('span').textContent = label;
    button.title = label;
    const use = button.querySelector('use');
    use.setAttribute('href', iconHref(use.getAttribute('href'), icon));
}

function setUpToggles(status) {
    const store = openListStore();
    const showStorageNote = () => {
        if (status && !store.persistent) {
            status.textContent = status.dataset.notSaved;
        }
    };
    const refreshers = [];

    for (const toggle of TOGGLES) {
        const button = document.querySelector(toggle.selector + '[data-movie-id]');
        if (button) {
            const movieId = Number(button.dataset.movieId);
            const refresh = () => setState(button, toggle, store.has(toggle.list, movieId));
            refresh();
            refreshers.push(refresh);
            button.addEventListener('click', () => {
                setState(button, toggle, store.toggle(toggle.list, movieId));
                // Storage can also fail on the first write, e.g. when it is full.
                showStorageNote();
            });
        }
    }
    // A button showing an old state would undo the other tab's change on the next click.
    watchLists(() => refreshers.forEach((refresh) => refresh()));
    showStorageNote();
}

// The thumbnail button becomes the YouTube player, which starts at once and takes the focus.
function setUpTrailer() {
    const trailer = document.querySelector('.trailer-play');
    trailer?.addEventListener('click', () => {
        const player = document.createElement('iframe');
        player.className = 'movie-trailer';
        player.src = `https://www.youtube-nocookie.com/embed/${encodeURIComponent(trailer.dataset.trailerKey)}?autoplay=1`;
        player.title = trailer.dataset.trailerTitle;
        player.allow = PLAYER_PERMISSIONS;
        player.allowFullscreen = true;
        trailer.replaceWith(player);
        player.focus();
    });
}

// The node tests import toggleView and iconHref without a page around them.
if (typeof document !== 'undefined') {
    const status = document.querySelector('.action-status');
    setUpToggles(status);
    setUpTrailer();
    // Set up, so base.js need not say that the buttons failed to load.
    status?.removeAttribute('data-load-error');
}
