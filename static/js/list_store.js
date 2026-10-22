// Favourites and watchlist, kept in this browser's localStorage; the server stores no lists.
// Each list is stored as a JSON array of movie ids, newest first, under its own key.
const LIST_NAMES = ['favorites', 'watchlist'];
// Earlier versions of the site used this key prefix; createListStore moves those lists.
const OLD_KEY_PREFIX = 'film-oneri:';
// The server's movie id range (MAX_MOVIE_ID in movierec/web/__init__.py).
const MAX_MOVIE_ID = 2147483647;

const isMovieId = (id) => Number.isInteger(id) && id > 0 && id <= MAX_MOVIE_ID;

function keyFor(name) {
    if (!LIST_NAMES.includes(name)) {
        throw new TypeError(`Unknown list: ${name}`);
    }
    return `movierec:${name}`;
}

function parse(raw) {
    try {
        const ids = JSON.parse(raw);
        return Array.isArray(ids) ? [...new Set(ids.filter(isMovieId))] : [];
    } catch {
        // A corrupt value starts the list afresh rather than breaking the page.
        return [];
    }
}

/**
 * A list store over a Storage such as localStorage. When storage is missing or throws
 * (blocked, full, private mode) the lists live in memory for this page and `persistent`
 * turns false so the page can say so (its data-not-saved text).
 */
export function createListStore(storage) {
    let persistent = Boolean(storage);
    const memory = new Map(LIST_NAMES.map((name) => [name, []]));

    // Read through on every call, so a change made in another tab is not overwritten.
    function read(name) {
        const key = keyFor(name);
        if (persistent) {
            try {
                memory.set(name, parse(storage.getItem(key)));
            } catch {
                persistent = false;
            }
        }
        return memory.get(name);
    }

    function write(name, ids) {
        memory.set(name, ids);
        if (!persistent) {
            return;
        }
        try {
            storage.setItem(keyFor(name), JSON.stringify(ids));
        } catch {
            persistent = false;
        }
    }

    // An old key's ids join the list after its own, without duplicates. The old key goes only
    // once the merged list is saved, so a failing storage loses nothing and the next visit tries
    // again.
    function migrate(name) {
        const ids = read(name);
        const oldKey = OLD_KEY_PREFIX + name;
        let old = null;
        if (persistent) {
            try {
                old = storage.getItem(oldKey);
            } catch {
                persistent = false;
            }
        }
        if (old === null) {
            return;
        }
        write(name, [...new Set([...ids, ...parse(old)])]);
        if (persistent) {
            try {
                storage.removeItem(oldKey);
            } catch {
                // The merge repeats without duplicates on the next visit.
            }
        }
    }

    // Both lists are read once, so memory holds them if storage fails later.
    LIST_NAMES.forEach(migrate);

    return {
        get persistent() {
            return persistent;
        },
        list: (name) => [...read(name)],
        has: (name, id) => read(name).includes(id),
        toggle(name, id) {
            if (!isMovieId(id)) {
                throw new TypeError(`Invalid movie id: ${id}`);
            }
            const ids = read(name);
            const active = !ids.includes(id);
            write(name, active ? [id, ...ids] : ids.filter((other) => other !== id));
            return active;
        },
    };
}

/** The store over this browser's localStorage; reading the property throws when it is blocked. */
export function openListStore() {
    let storage = null;
    try {
        storage = window.localStorage;
    } catch {
        // createListStore falls back to memory.
    }
    return createListStore(storage);
}

/**
 * Calls back when the lists may have changed outside this page: in another tab (the storage
 * event) or while the page waited in the back/forward cache (pageshow). The store reads
 * through, so the page only has to read it again.
 */
export function watchLists(callback, target = window) {
    const keys = LIST_NAMES.map(keyFor);
    target.addEventListener('storage', (event) => {
        // A null key means another tab cleared the storage.
        if (event.key === null || keys.includes(event.key)) {
            callback();
        }
    });
    target.addEventListener('pageshow', (event) => {
        if (event.persisted) {
            callback();
        }
    });
}
