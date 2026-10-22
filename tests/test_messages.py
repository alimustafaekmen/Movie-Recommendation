"""The message catalogue, movierec/messages.json: its shape, and that the code and it agree."""

import json
import re
import string
from html.parser import HTMLParser
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
# Text that stays the same in both languages: the brand (and the logo's two halves), the codes
# and names of the language switch, and TMDB's name and the notice its terms ask for in English.
SAME_IN_BOTH = {
    "MovieRec",
    "Movie",
    "Rec",
    "TR",
    "EN",
    "– Türkçe",
    "– English",
    "The Movie Database (TMDB)",
    "This website uses TMDB and the TMDB APIs but is not endorsed, certified,"
    " or otherwise approved by TMDB.",
}
# Attributes people read or hear; every data-* attribute counts too, except these names.
TEXT_ATTRIBUTES = {"aria-label", "title", "alt", "placeholder"}
NAME_ATTRIBUTES = {"data-dismiss"}
JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.S)
JINJA_CODE = re.compile(r"\{\{.*?\}\}|\{%.*?%\}", re.S)
# A string in Jinja code that is a name, not text: a key, a route, a class, an icon, a value.
CODE_STRING = re.compile(r"[a-z0-9_./#:+ -]*")
TURKISH_LETTERS = re.compile("[çğıİöşüÇĞÖŞÜ]")


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


class _TextCollector(HTMLParser):
    """The text nodes and the text attribute values of a page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.found = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            is_data = name.startswith("data-") and name not in NAME_ATTRIBUTES
            if value and (name in TEXT_ATTRIBUTES or is_data):
                self.found.append(value)

    def handle_data(self, data):
        self.found.append(data)


def _template_texts(source):
    """The literal text of a template: its text nodes and text attributes, and the strings in its
    Jinja code that are not names."""
    source = JINJA_COMMENT.sub("", source)
    # The body's class is a name for the stylesheets.
    source = re.sub(r"\{% block body_class %\}.*?\{% endblock %\}", "", source)
    found = []
    for code in JINJA_CODE.findall(source):
        strings = re.findall(r"""(["'])(.*?)\1""", code)
        found += [text for _, text in strings if not CODE_STRING.fullmatch(text)]
    # The brand's tag name is Jinja code (h1 or span); everything else drops out, and a number
    # stands in for an expression, so no letters are left where a t() call was.
    source = re.sub(r"<(/?)\{\{.*?\}\}", r"<\1span", source)
    source = re.sub(r"\{\{.*?\}\}", "0", source, flags=re.S)
    source = re.sub(r"\{%.*?%\}", "", source, flags=re.S)
    collector = _TextCollector()
    collector.feed(source)
    collector.close()
    return found + collector.found


def _untranslated(source):
    """The texts of a template with letters that do not go through the catalogue."""
    found = []
    for text in _template_texts(source):
        # A page title built from data is the data, then the brand.
        text = " ".join(text.split()).removesuffix(" · MovieRec")
        if any(char.isalpha() for char in text) and text not in SAME_IN_BOTH:
            found.append(text)
    return found


@pytest.mark.parametrize("template", sorted(p.name for p in (REPO / "templates").glob("*.html")))
def test_templates_write_every_text_through_the_catalogue(template):
    source = (REPO / "templates" / template).read_text(encoding="utf-8")

    assert _untranslated(source) == []


def test_the_untranslated_check_finds_a_forgotten_text():
    source = """<p>{{ t('home.intro') }}</p><img alt="Poster"><b data-label="{{ t('x.y') }}">0</b>
    {{ movie_card(movie, note_label="Rol") }}<a title="{{ t('x.y') }}">Hepsi</a>"""

    assert _untranslated(source) == ["Rol", "Poster", "Hepsi"]


@pytest.mark.parametrize("script", sorted(p.name for p in (REPO / "static" / "js").glob("*.js")))
def test_scripts_get_their_texts_from_the_page(script):
    code = (REPO / "static" / "js" / script).read_text(encoding="utf-8")
    # Comments may quote the Turkish texts; "https://" is not a comment.
    code = re.sub(r"/\*.*?\*/|(?<![:\w])//[^\n]*", "", code, flags=re.S)

    assert not TURKISH_LETTERS.findall(code)
