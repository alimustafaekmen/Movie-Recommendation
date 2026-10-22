"""The message catalogue, movierec/messages.json: its shape, and that the code and it agree."""

import json
import re
import string
from pathlib import Path

import pytest

from movierec.i18n import LANGUAGES, MESSAGES

REPO = Path(__file__).resolve().parents[1]
CATALOGUE = REPO / "movierec" / "messages.json"
SOURCES = [
    *sorted((REPO / "templates").glob("*.html")),
    *sorted((REPO / "movierec").rglob("*.py")),
    *sorted((REPO / "static" / "js").glob("*.js")),
]
PLURAL_FORMS = ("one", "other")
# A key as the code writes it: in quotes, in full, never built from parts.
QUOTED_KEY = re.compile(r"""["']([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)["']""")
# Calls with the key as the first argument, and those with the language first.
KEY_CALL = re.compile(r"""\b(t|t_count|message)\(\s*["']([^"']*)["']""")
LANG_KEY_CALL = re.compile(r"""\b(translate|translate_count)\(\s*\w+,\s*["']([^"']*)["']""")
PLURAL_CALLS = {"t_count", "translate_count"}


def _placeholders(text):
    return {field for _, field, _, _ in string.Formatter().parse(text) if field is not None}


def _source_text():
    return "\n".join(path.read_text(encoding="utf-8") for path in SOURCES)


def _plural_base(key):
    base, _, form = key.rpartition(".")
    return base if form in PLURAL_FORMS else None


def test_the_file_is_sorted_and_formatted():
    text = CATALOGUE.read_text(encoding="utf-8")

    assert text == json.dumps(MESSAGES, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize("key", sorted(MESSAGES))
def test_every_key_holds_both_languages_with_the_same_placeholders(key):
    texts = MESSAGES[key]

    assert re.fullmatch(r"[a-z]+(\.[a-z0-9_]+)+", key)
    assert sorted(texts) == sorted(LANGUAGES)
    assert all(text.strip() for text in texts.values())
    assert _placeholders(texts["tr"]) == _placeholders(texts["en"])
    # Plain text, so it is safe as element text and inside a double-quoted attribute.
    assert not [text for text in texts.values() if re.search(r'[<>&"]', text)]


def test_plural_keys_come_in_pairs():
    bases = {_plural_base(key) for key in MESSAGES} - {None}

    for base in bases:
        assert {f"{base}.{form}" for form in PLURAL_FORMS} <= set(MESSAGES), base


def test_every_key_is_used():
    used = set(QUOTED_KEY.findall(_source_text()))

    unused = [key for key in MESSAGES if key not in used and _plural_base(key) not in used]
    assert unused == []


def test_every_key_the_code_asks_for_exists():
    source = _source_text()
    calls = [*KEY_CALL.findall(source), *LANG_KEY_CALL.findall(source)]

    assert calls
    for function, key in calls:
        if function in PLURAL_CALLS:
            assert f"{key}.one" in MESSAGES, (function, key)
        else:
            assert key in MESSAGES, (function, key)
