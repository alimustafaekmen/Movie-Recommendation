// Like and watchlist toggles on the movie detail page. The lists live in this browser,
// so the server renders both buttons inactive and this module sets their real state.
import { NOT_SAVED_MESSAGE, openListStore } from './lists.js';

const TOGGLES = [
    { selector: '.btn-like', list: 'favorites', activeClass: 'liked', label: 'Beğen', activeLabel: 'Beğenildi' },
    { selector: '.btn-watchlist', list: 'watchlist', activeClass: 'added', label: 'Daha Sonra İzle', activeLabel: 'Listede' },
];

const store = openListStore();
const status = document.querySelector('.action-status');

function showStorageNote() {
    if (status && !store.persistent) {
        status.textContent = NOT_SAVED_MESSAGE;
    }
}

function setState(button, toggle, active) {
    button.classList.toggle(toggle.activeClass, active);
    button.setAttribute('aria-pressed', String(active));
    button.querySelector('span').textContent = active ? toggle.activeLabel : toggle.label;
}

for (const toggle of TOGGLES) {
    const button = document.querySelector(toggle.selector + '[data-movie-id]');
    if (button) {
        const movieId = Number(button.dataset.movieId);
        setState(button, toggle, store.has(toggle.list, movieId));
        button.addEventListener('click', () => {
            setState(button, toggle, store.toggle(toggle.list, movieId));
            // Storage can also fail on the first write, e.g. when it is full.
            showStorageNote();
        });
    }
}
showStorageNote();
