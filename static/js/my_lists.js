// The my lists page: the ids come from this browser's storage, the cards from /api/movies and
// the films like the favourites from /api/recommendations.
import { NOT_SAVED_MESSAGE, openListStore, watchLists } from './list_store.js';

// The most ids /api/recommendations reads per list.
const MAX_IDS_PER_REQUEST = 100;
// /api/movies looks up at most this many ids outside the dataset (MAX_TMDB_LOOKUPS), so chunks
// of this size never lose one.
const MAX_CARD_IDS_PER_REQUEST = 20;
const LOAD_FAILED_MESSAGE = 'Filmler şu anda yüklenemedi. Lütfen sayfayı yenileyin.';
const SOME_FAILED_MESSAGE = 'Bazı filmler şu anda yüklenemedi.';

// Plain code-point order, so the lists sort the same whatever the browser's language.
const byTitle = (a, b) => (a.title < b.title ? -1 : a.title > b.title ? 1 : 0);

/**
 * What the page shows for the stored lists (arrays of ids) once their cards (a Map by id) have
 * loaded: each list's cards in title order, and the status line.
 */
export function describeLists(lists, cards, persistent) {
    const sections = lists.map((ids) => ids.map((id) => cards.get(id)).filter(Boolean).sort(byTitle));
    // /api/movies leaves out ids it could not resolve, e.g. during a TMDb outage.
    const incomplete = sections.some((shown, index) => shown.length < lists[index].length);
    const status = [persistent ? '' : NOT_SAVED_MESSAGE, incomplete ? SOME_FAILED_MESSAGE : ''];
    return { sections, status: status.filter(Boolean).join(' ') };
}

// What renderCard needs; the other fields are checked where they are shown.
const isCard = (card) =>
    typeof card === 'object' && card !== null && Number.isInteger(card.id) && typeof card.title === 'string';

/**
 * The cards (a Map by id) for the ids from the cards URL, MAX_CARD_IDS_PER_REQUEST ids per request.
 * A request that fails (network, error status, a body that is not a JSON list) only loses its
 * own cards and sets failed; malformed cards are left out like ids the server could not resolve.
 */
export async function fetchCards(url, ids) {
    const cards = new Map();
    let failed = false;
    for (let start = 0; start < ids.length; start += MAX_CARD_IDS_PER_REQUEST) {
        const query = ids.slice(start, start + MAX_CARD_IDS_PER_REQUEST).join(',');
        try {
            const response = await fetch(`${url}?ids=${query}`, {
                headers: { Accept: 'application/json' },
            });
            if (!response.ok) {
                throw new Error(`Unexpected response ${response.status}`);
            }
            const body = await response.json();
            if (!Array.isArray(body)) {
                throw new TypeError('Expected a list of cards');
            }
            body.filter(isCard).forEach((card) => cards.set(card.id, card));
        } catch {
            failed = true;
        }
    }
    return { cards, failed };
}

/**
 * The cards from the recommendations URL for the favourites and watchlist, of which it sends the
 * first MAX_IDS_PER_REQUEST ids each. No favourites asks for nothing; a failed request gives no
 * cards, so the section stays hidden.
 */
export async function fetchRecommendations(url, favorites, watchlist) {
    if (!favorites.length) {
        return [];
    }
    const query = (ids) => ids.slice(0, MAX_IDS_PER_REQUEST).join(',');
    try {
        const response = await fetch(`${url}?favorites=${query(favorites)}&watchlist=${query(watchlist)}`, {
            headers: { Accept: 'application/json' },
        });
        if (!response.ok) {
            throw new Error(`Unexpected response ${response.status}`);
        }
        const body = await response.json();
        return Array.isArray(body) ? body.filter(isCard) : [];
    } catch {
        return [];
    }
}

/**
 * The recommended cards to show, in the server's order: none without favourites, and never a
 * film on a list, as the server only reads the first MAX_IDS_PER_REQUEST ids of each.
 */
export function recommendationsToShow(favorites, watchlist, cards) {
    const listed = new Set([...favorites, ...watchlist]);
    return favorites.length ? cards.filter((card) => !listed.has(card.id)) : [];
}

/** A rating such as 8.2 as '8,2', as the format_rating filter shows it; no rating gives ''. */
export function formatRating(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || !value) {
        return '';
    }
    return value.toLocaleString('tr-TR', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
}

/**
 * A movie page URL from the page's data-movie-url, the server's URL for id 0: the id takes the
 * place of that 0 (the last digit), so the routes stay the server's business.
 */
export function movieHref(movieUrl, id) {
    return movieUrl.replace(/0(?=\D*$)/, String(id));
}

/** What the movie card (templates/_movie_card.html) shows for an /api/movies card. */
export function cardContent(card, movieUrl) {
    return {
        href: movieHref(movieUrl, card.id),
        title: card.title,
        // A plain year, never '2.008'; 0 means unknown.
        year: Number.isInteger(card.year) && card.year > 0 ? String(card.year) : '',
        rating: formatRating(card.vote_average),
        poster: typeof card.poster_url === 'string' ? card.poster_url : '',
    };
}

// Adds the text to a part of the card, or removes the part when there is none, as the macro does.
function fill(element, text) {
    if (text) {
        element.append(text);
    } else {
        element.remove();
    }
}

// A card while its film loads: the placeholder poster and an empty body, as tall as a real card.
function renderPlaceholder(template) {
    const item = template.content.firstElementChild.cloneNode(true);
    item.querySelector('.movie-card-poster').remove();
    item.querySelector('.movie-card-meta').replaceChildren();
    return item;
}

function renderCard(template, card, movieUrl) {
    const shown = cardContent(card, movieUrl);
    const item = template.content.firstElementChild.cloneNode(true);
    item.querySelector('.movie-card').href = shown.href;
    const poster = item.querySelector('.movie-card-poster');
    if (shown.poster) {
        poster.src = shown.poster;
        item.querySelector('.movie-card-placeholder').remove();
    } else {
        poster.remove();
    }
    item.querySelector('.movie-card-title').textContent = shown.title;
    fill(item.querySelector('.movie-card-year'), shown.year);
    fill(item.querySelector('.movie-card-rating'), shown.rating && ` ${shown.rating}`);
    return item;
}

// A list with movies gets its heading text wrapped in the collapse button that base.js
// handles for every page, with the chevron from the page's template after the text; a list that
// is empty again gets its plain heading back.
function setToggle(section, grid, on) {
    const heading = section.querySelector('h2');
    const button = heading.querySelector('.list-toggle');
    if (on && !button) {
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'list-toggle';
        toggle.setAttribute('aria-expanded', 'true');
        toggle.setAttribute('aria-controls', grid.id);
        const chevron = document.getElementById('list-toggle-chevron-template').content.firstElementChild.cloneNode(true);
        toggle.append(...heading.childNodes, chevron);
        heading.append(toggle);
    } else if (!on && button) {
        button.querySelector('.list-toggle-chevron').remove();
        button.replaceWith(...button.childNodes);
    }
}

// One list: the empty message when nothing is stored, else a placeholder per stored film while
// its cards load (cards is null), else the loaded cards. Only stored ids decide emptiness, so a
// list whose films fail to load is never shown as empty. A list the visitor folded stays folded.
function showList(section, count, cards, template, movieUrl) {
    const grid = section.querySelector('.movie-grid');
    section.querySelector('.empty-list').hidden = count > 0;
    if (cards) {
        grid.replaceChildren(...cards.map((card) => renderCard(template, card, movieUrl)));
        grid.removeAttribute('aria-busy');
        setToggle(section, grid, cards.length > 0);
    } else {
        grid.replaceChildren(...Array.from({ length: count }, () => renderPlaceholder(template)));
        grid.setAttribute('aria-busy', 'true');
    }
    const folded = section.querySelector('.list-toggle')?.getAttribute('aria-expanded') === 'false';
    grid.hidden = !grid.children.length || folded;
}

function main() {
    const page = document.querySelector('.my-lists-page');
    const { cardsUrl, movieUrl, recommendationsUrl } = page.dataset;
    const status = page.querySelector('.list-status');
    const template = document.getElementById('movie-card-template');
    const store = openListStore();
    const sections = [...page.querySelectorAll('[data-list]')];
    // Cards fetched so far, so drawing the lists again only asks for films added since.
    const known = new Map();
    let renders = 0;
    let drawn = null;
    const recommendations = page.querySelector('.recommendations');
    let recommendationRuns = 0;

    // The section keeps its cards until the new ones arrive, so a list change does not make it
    // blink; it hides when there is nothing to show.
    async function showRecommendations(favorites, watchlist) {
        const run = ++recommendationRuns;
        const cards = await fetchRecommendations(recommendationsUrl, favorites, watchlist);
        if (run !== recommendationRuns) {
            return;
        }
        const shown = recommendationsToShow(favorites, watchlist, cards);
        recommendations.querySelector('.movie-grid').replaceChildren(...shown.map((card) => renderCard(template, card, movieUrl)));
        recommendations.hidden = !shown.length;
    }

    async function render() {
        const run = ++renders;
        const lists = sections.map((section) => store.list(section.dataset.list));
        const missing = [...new Set(lists.flat())].filter((id) => !known.has(id));
        if (!missing.length && JSON.stringify(lists) === drawn) {
            return;
        }
        showRecommendations(store.list('favorites'), store.list('watchlist'));
        let failed = false;
        if (missing.length) {
            lists.forEach((ids, index) => showList(sections[index], ids.length, ids.length ? null : [], template));
            status.textContent = store.persistent ? '' : NOT_SAVED_MESSAGE;
            // The placeholders replaced what was drawn, even if this render is overtaken.
            drawn = null;
            const loaded = await fetchCards(cardsUrl, missing);
            loaded.cards.forEach((card, id) => known.set(id, card));
            // The lists changed again meanwhile; the newer render draws them.
            if (run !== renders) {
                return;
            }
            failed = loaded.failed;
        }
        const shown = describeLists(lists, known, store.persistent);
        const nothingLoaded = shown.sections.every((cards) => !cards.length);
        status.textContent = failed && nothingLoaded ? LOAD_FAILED_MESSAGE : shown.status;
        sections.forEach((section, index) => showList(section, lists[index].length, shown.sections[index], template, movieUrl));
        drawn = JSON.stringify(lists);
    }

    render();
    watchLists(render);
    // The module is running, so base.js need not say that the page failed to load.
    status.removeAttribute('data-load-error');
}

// The node tests import this module without a page around it.
if (typeof document !== 'undefined') {
    main();
}
