// The my lists page: the ids come from this browser's storage, the cards from /api/movies.
import { NOT_SAVED_MESSAGE, openListStore } from './lists.js';

// The most ids /api/movies accepts in one request.
const CHUNK_SIZE = 100;
const LOADING_MESSAGE = 'Filmler yükleniyor…';
const ERROR_MESSAGE = 'Filmler şu anda yüklenemedi. Lütfen sayfayı yenileyin.';
const SOME_FAILED_MESSAGE = 'Bazı filmler şu anda yüklenemedi.';

// Plain code-point order, so the lists sort the same whatever the browser's language.
const byTitle = (a, b) => (a.title < b.title ? -1 : a.title > b.title ? 1 : 0);

/**
 * What the page shows for the stored lists (arrays of ids) once their cards (a Map by id) have
 * loaded: each list's cards in title order, whether it shows its empty message, and the status.
 */
export function describeLists(lists, cards, persistent) {
    const sections = lists.map((ids) => ({
        cards: ids.map((id) => cards.get(id)).filter(Boolean).sort(byTitle),
        // A list whose films failed to load is not empty; the status line says so instead.
        empty: ids.length === 0,
    }));
    // /api/movies leaves out ids it could not resolve, e.g. during a TMDb outage.
    const incomplete = sections.some((section, index) => section.cards.length < lists[index].length);
    const status = [persistent ? '' : NOT_SAVED_MESSAGE, incomplete ? SOME_FAILED_MESSAGE : ''];
    return { sections, status: status.filter(Boolean).join(' ') };
}

async function fetchCards(url, ids) {
    const cards = new Map();
    for (let start = 0; start < ids.length; start += CHUNK_SIZE) {
        const query = ids.slice(start, start + CHUNK_SIZE).join(',');
        const response = await fetch(`${url}?ids=${query}`, {
            headers: { Accept: 'application/json' },
        });
        if (!response.ok) {
            throw new Error('Unexpected response ' + response.status);
        }
        for (const card of await response.json()) {
            cards.set(card.id, card);
        }
    }
    return cards;
}

function renderCard(template, card) {
    const link = template.content.firstElementChild.cloneNode(true);
    link.href = `/movie/${card.id}`;
    link.title = card.title;
    const poster = link.querySelector('img');
    if (card.poster_url) {
        poster.src = card.poster_url;
        poster.alt = `${card.title} Posteri`;
        link.querySelector('.poster-placeholder').remove();
    } else {
        poster.remove();
    }
    link.querySelector('.movie-card-personal-title').textContent = card.title;
    const rating = link.querySelector('.movie-card-personal-rating');
    if (card.vote_average > 0) {
        rating.append(` ${card.vote_average.toFixed(1)}/10`);
    } else {
        rating.remove();
    }
    return link;
}

// Turns the heading into the collapse button that scripts.js handles for every page.
function addToggle(heading, grid) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'toggle-button';
    button.setAttribute('aria-expanded', 'true');
    button.setAttribute('aria-controls', grid.id);
    const icon = document.createElement('span');
    icon.className = 'toggle-icon';
    icon.innerHTML = '<i class="fas fa-chevron-down"></i>';
    button.append(...heading.childNodes, ' ', icon);
    heading.replaceChildren(button);
}

function renderSection(section, { cards, empty }, template) {
    if (empty) {
        section.querySelector('.empty-list-message').hidden = false;
    }
    if (!cards.length) {
        return;
    }
    const grid = section.querySelector('.movies-grid');
    grid.replaceChildren(...cards.map((card) => renderCard(template, card)));
    grid.hidden = false;
    addToggle(section.querySelector('h2'), grid);
}

async function main() {
    const page = document.querySelector('.personal-page');
    const status = page.querySelector('.list-status');
    const store = openListStore();
    const sections = [...page.querySelectorAll('[data-list]')];
    const lists = sections.map((section) => store.list(section.dataset.list));
    const ids = [...new Set(lists.flat())];
    let cards = new Map();
    if (ids.length) {
        status.textContent = LOADING_MESSAGE;
        try {
            cards = await fetchCards(page.dataset.cardsUrl, ids);
        } catch {
            status.textContent = ERROR_MESSAGE;
            return;
        }
    }
    const shown = describeLists(lists, cards, store.persistent);
    status.textContent = shown.status;
    const template = document.getElementById('movie-card-template');
    sections.forEach((section, index) => renderSection(section, shown.sections[index], template));
}

// The node tests import describeLists without a page around it.
if (typeof document !== 'undefined') {
    main();
}
