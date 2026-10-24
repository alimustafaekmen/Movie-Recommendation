"""Shared layout of every page: landmarks, header, footer, flashes, scripts and static assets."""

import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree

import pytest
from conftest import SEARCHES
from flask import render_template_string
from pages import Page, body_text, texts

from movierec.web import build_asset_versions

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "templates"
STATIC = REPO / "static"
BASE_CSS = STATIC / "css" / "base.css"
SPRITE = STATIC / "icons.svg"
SVG = "{http://www.w3.org/2000/svg}"
FONT_AWESOME_LICENCE = (
    "Font Awesome Free 6.5.1 by @fontawesome - https://fontawesome.com"
    " License - https://fontawesome.com/license/free"
)
SITE_NAME = "MovieRec"
# The exact notice section 3 of TMDB's API terms asks for, in English as written there.
TMDB_NOTICE = (
    "This website uses TMDB and the TMDB APIs but is not endorsed, certified,"
    " or otherwise approved by TMDB."
)
HOME_TITLE = "MovieRec · Benzer filmleri keşfedin"
# The site's name before the rebrand, written so this file does not match a grep for it.
OLD_SITE_NAME = re.compile(r"film\s+öneri\s+motoru", re.IGNORECASE)
FAVICON = STATIC / "favicon.svg"
FLASH_TEXT = "Deneme bildirimi"
# The stylesheet each page type loads after css/base.css and css/cards.css.
PAGE_STYLESHEETS = {
    "home": ["css/home.css"],
    "genre": ["css/home.css"],
    "results": ["css/home.css"],
    "movie": ["css/detail.css"],
    "person": ["css/detail.css"],
    "my_lists": ["css/lists.css"],
    "not_found": [],
}
BREAKPOINTS = ("600px", "900px", "1200px")
# The token blocks of css/base.css: the light theme, then the dark one twice, once for the
# system preference and once for the visitor's choice.
LIGHT_BLOCK = ("", ":root")
DARK_SYSTEM_BLOCK = ("@media (prefers-color-scheme: dark)", ':root:not([data-theme="light"])')
DARK_CHOICE_BLOCK = ("", ':root[data-theme="dark"]')
# Colour tokens that are the same in both themes: the header is dark in both, and the play
# button sits on a video frame.
THEME_INDEPENDENT_COLOURS = {
    "--color-header-text",
    "--color-header-text-hover",
    "--color-header-muted",
    "--color-header-hover-overlay",
    "--color-header-field",
    "--color-header-field-border",
    "--color-header-field-border-hover",
    "--color-header-hairline",
    "--color-play-bg",
    "--color-play-bg-hover",
    "--color-play-icon",
}
# A colour written out: hex (also %23-escaped in a data URL), a colour function or a name.
COLOUR_LITERAL = re.compile(
    r"#[0-9a-f]{3,8}\b|%23[0-9a-f]{3,8}\b"
    r"|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color|color-mix)\("
    r"|\b(?:white|black|red|green|blue|gray|grey|silver|yellow|orange|purple|pink|navy)\b",
    re.IGNORECASE,
)
STYLESHEETS = sorted(p.name for p in (STATIC / "css").glob("*.css"))
THEMES = ("light", "dark")


def _css_rules(css):
    """(at-rule, selector, declarations) of every style rule; at-rule is "" at the top level."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    at_rules, selector, start = [], None, 0
    for brace in re.finditer(r"[{}]", css):
        text = " ".join(css[start : brace.start()].split())
        start = brace.end()
        if brace.group() == "{" and text.startswith("@"):
            at_rules.append(text)
        elif brace.group() == "{":
            selector = text
        elif selector is not None:
            yield (at_rules[-1] if at_rules else "", selector, text)
            selector = None
        else:
            at_rules.pop()


def _declarations(sheet, selector, at_rule=""):
    """The declarations of `selector` in `sheet` (in `at_rule`), joined into one string."""
    css = sheet.read_text(encoding="utf-8")
    return " ".join(
        body
        for media, selectors, body in _css_rules(css)
        if media == at_rule and selector in (part.strip() for part in selectors.split(","))
    )


def test_every_page_has_one_h1_and_one_main_landmark(pages):
    for name, body in pages.items():
        page = Page(body)
        assert len(page.all("h1")) == 1, name
        assert [main.get("id") for main in page.all("main")] == ["main-content"], name


def test_footer_carries_the_site_name_and_the_tmdb_notice(client):
    [footer] = Page(body_text(client.get("/"))).find("footer")

    assert SITE_NAME in footer.text
    assert texts(footer, lang="en") == [TMDB_NOTICE]
    assert footer.find("a", href="https://www.themoviedb.org/")


def _render_page(app, blocks):
    """A page that extends base.html with the given block overrides."""
    with app.test_request_context("/"):
        return render_template_string('{% extends "base.html" %}' + blocks)


def test_a_page_can_set_a_class_on_the_body(app_module):
    body = _render_page(app_module.app, "{% block body_class %}page-test{% endblock %}")

    [tag] = Page(body).all("body")
    assert tag["class"] == "page-test"


# Classes every page can use, so the page stylesheets do not each restyle them.
SHARED_CLASSES = [
    ".page",
    ".page-title",
    ".section-title",
    ".btn",
    ".btn-primary",
    ".btn-secondary",
    ".tag-list",
    ".tag",
    ".meta-line",
    ".panel",
    ".link",
    ".sr-only",
]


@pytest.mark.parametrize("selector", SHARED_CLASSES)
def test_shared_classes_are_styled_in_base_css(selector):
    assert _declarations(BASE_CSS, selector)


def test_page_titles_use_the_condensed_display_face_and_everything_else_is_sans():
    assert "font-family: var(--font-display)" in _declarations(BASE_CSS, ".page-title")
    assert "font-family: var(--font-sans)" in _declarations(BASE_CSS, "body")
    css = BASE_CSS.read_text(encoding="utf-8")
    assert re.search(r'--font-sans:\s*"IBM Plex Sans",', css)
    assert re.search(r'--font-display:\s*"IBM Plex Sans Condensed", "IBM Plex Sans",', css)


@pytest.mark.parametrize("sheet", STYLESHEETS)
def test_stylesheets_name_only_the_two_plex_families(sheet):
    css = (STATIC / "css" / sheet).read_text(encoding="utf-8")

    # Every font-family goes through the two tokens (the @font-face rules that name the files
    # are no style rules); the old faces and token are gone.
    for _, _, body in _css_rules(css):
        for family in re.findall(r"font-family:\s*([^;]+)", body):
            assert family.strip() in {"var(--font-sans)", "var(--font-display)", "inherit"}, sheet
    for old in ("Inter", "Source Serif", "--font-serif", "Poppins", "Roboto Slab"):
        assert old not in css, (sheet, old)


def test_primary_buttons_are_ink_with_the_text_colour_that_flips_with_it():
    # Primary actions are ink, not the accent; in the dark theme the ink is light, so the
    # text on it must be the on-ink token, never a fixed white.
    rule = _declarations(BASE_CSS, ".btn-primary")

    assert "background-color: var(--color-ink)" in rule
    assert "color: var(--color-on-ink)" in rule


def test_meta_line_separators_are_not_read_aloud():
    # The middle dot is decoration: the alternative text after the slash is empty.
    [rule] = [
        body
        for _, selector, body in _css_rules(BASE_CSS.read_text(encoding="utf-8"))
        if selector.startswith(".meta-line") and "::before" in selector
    ]
    assert re.search(r'content:\s*"\\00B7"\s*/\s*""', rule)


def test_the_hidden_attribute_beats_any_display_rule():
    assert "display: none !important" in _declarations(BASE_CSS, "[hidden]")


def test_home_page_hides_the_header_search_below_900px():
    # The home page has its own search panel; two search boxes on a phone compete.
    rule = ".page-home .header-search-form"

    assert "display: none" in _declarations(BASE_CSS, rule)
    assert "display: flex" in _declarations(BASE_CSS, rule, "@media (min-width: 900px)")


def _token_blocks():
    """{(at-rule, selector): {property: value}} of the :root rules of css/base.css."""
    blocks = {}
    for at_rule, selector, body in _css_rules(BASE_CSS.read_text(encoding="utf-8")):
        if selector.startswith(":root"):
            declarations = (part.partition(":") for part in body.split(";"))
            blocks[(at_rule, selector)] = {
                name.strip(): value.strip() for name, _, value in declarations if value.strip()
            }
    return blocks


def _tokens(theme="light"):
    """{token: value} of a theme: the light tokens, with the dark block over them."""
    blocks = _token_blocks()
    tokens = dict(blocks[LIGHT_BLOCK])
    if theme == "dark":
        tokens.update(blocks[DARK_CHOICE_BLOCK])
    return tokens


def _contrast(first, second):
    """WCAG contrast ratio of two #RRGGBB colours."""

    def luminance(colour):
        channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


# WCAG AA: 4.5:1 for text, 3:1 for graphics and the borders that show where a control is.
@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize(
    ("foreground", "background", "minimum"),
    [
        ("--color-ink", "--color-bg", 4.5),
        ("--color-ink", "--color-surface", 4.5),
        ("--color-ink", "--color-tag-bg-hover", 4.5),
        ("--color-text-muted", "--color-surface", 4.5),
        ("--color-text-muted", "--color-bg", 4.5),
        ("--color-text-muted", "--color-tag-bg", 4.5),
        # The poster placeholders' "No poster" sits on the quiet fill.
        ("--color-text-muted", "--color-subtle", 4.5),
        ("--color-text-muted", "--color-accent-soft", 4.5),
        ("--color-accent", "--color-surface", 4.5),
        ("--color-accent", "--color-bg", 4.5),
        ("--color-accent-hover", "--color-bg", 4.5),
        ("--color-on-ink", "--color-ink", 4.5),
        ("--color-on-ink", "--color-ink-hover", 4.5),
        ("--color-control-border", "--color-surface", 3),
        ("--color-control-border", "--color-bg", 3),
        ("--color-rating", "--color-bg", 3),
        ("--color-rating", "--color-surface", 3),
        ("--color-header-text", "--color-header", 4.5),
        ("--color-header-muted", "--color-header", 4.5),
        ("--color-header-muted", "--color-header-field", 4.5),
        ("--color-header-text", "--color-header-field", 4.5),
        # The accent rule under the current nav link, and the logo's dot.
        ("--color-accent", "--color-header", 3),
        ("--color-warning-text", "--color-warning-bg", 4.5),
        ("--color-danger-text", "--color-danger-bg", 4.5),
        # The danger text also outlines an invalid field.
        ("--color-danger-text", "--color-surface", 3),
    ],
)
def test_colour_tokens_meet_wcag_aa_contrast(theme, foreground, background, minimum):
    tokens = _tokens(theme)

    assert _contrast(tokens[foreground], tokens[background]) >= minimum, (theme, tokens[foreground])


def test_the_two_dark_blocks_are_identical_and_override_every_themed_colour():
    blocks = _token_blocks()
    light, system, choice = (blocks[b] for b in (LIGHT_BLOCK, DARK_SYSTEM_BLOCK, DARK_CHOICE_BLOCK))
    colours = {name for name, value in light.items() if COLOUR_LITERAL.search(value)}

    # Copies, in the same order, so the system preference and the visitor's choice look alike.
    assert list(system.items()) == list(choice.items())
    # Native controls and scrollbars follow the theme.
    assert (light["color-scheme"], choice["color-scheme"]) == ("light", "dark")
    # The dark theme only re-colours: no new token, and every colour token changes except
    # the ones that are the same in both themes.
    assert set(choice) - {"color-scheme"} <= colours
    assert colours - set(choice) == THEME_INDEPENDENT_COLOURS
    assert [name for name in choice if name != "color-scheme" and choice[name] == light[name]] == []


@pytest.mark.parametrize("sheet", STYLESHEETS)
def test_colours_are_written_only_in_the_token_blocks(sheet):
    # Every colour comes from a token, so the dark theme only has to re-colour the tokens.
    path = STATIC / "css" / sheet
    token_blocks = set(_token_blocks()) if path == BASE_CSS else set()

    for at_rule, selector, body in _css_rules(path.read_text(encoding="utf-8")):
        if (at_rule, selector) in token_blocks:
            continue
        for declaration in body.split(";"):
            # Token names may contain a colour word (--color-red); only the rest is checked.
            value = re.sub(r"var\([^)]*\)", "", declaration.partition(":")[2])
            assert not COLOUR_LITERAL.search(value), (sheet, selector, declaration.strip())


def test_brand_palette_is_white_ink_and_one_red():
    tokens = _tokens()

    assert {
        name: tokens[name].upper()
        for name in (
            "--color-bg",
            "--color-surface",
            "--color-ink",
            "--color-header",
            "--color-accent",
        )
    } == {
        "--color-bg": "#FFFFFF",
        "--color-surface": "#FFFFFF",
        "--color-ink": "#0E0E10",
        "--color-header": "#0E0E10",
        "--color-accent": "#D7261E",
    }


# The greys of the page and the header; the alerts and the accent are the only hues.
GREY_TOKENS = [
    "--color-subtle",
    "--color-hairline",
    "--color-hairline-strong",
    "--color-control-border",
    "--color-tag-bg",
    "--color-tag-bg-hover",
    "--color-text-muted",
    "--color-ink-hover",
    "--color-header-text",
    "--color-header-muted",
    "--color-header-field",
    "--color-header-field-border",
    "--color-header-field-border-hover",
]


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("token", GREY_TOKENS)
def test_greys_are_neutral_without_a_warm_tint(theme, token):
    colour = _tokens(theme)[token]
    red, green, blue = (int(colour[i : i + 2], 16) for i in (1, 3, 5))

    assert max(red, green, blue) - min(red, green, blue) <= 10, (token, colour)
    # Warm greys lean red/yellow; these may only lean cool, like the ink.
    assert red <= blue and green <= blue, (token, colour)


def test_page_titles_end_with_the_site_name(pages):
    for name in ("movie", "person", "my_lists", "not_found"):
        [title] = re.findall(r"<title>(.*?)</title>", pages[name], re.S)
        assert re.fullmatch(rf"\S.* · {SITE_NAME}", " ".join(title.split())), (name, title)


def test_home_page_and_its_searches_lead_with_the_site_name(pages):
    for name in ("home", "genre", "results"):
        [title] = re.findall(r"<title>(.*?)</title>", pages[name], re.S)
        assert " ".join(title.split()) == HOME_TITLE, name


@pytest.mark.parametrize(
    ("method", "path", "heading"),
    [("get", "/does-not-exist", "Sayfa bulunamadı"), ("put", "/", "Bir sorun oluştu")],
)
def test_error_pages_name_the_problem_in_the_heading_and_title(client, method, path, heading):
    page = Page(body_text(getattr(client, method)(path)))

    [h1] = page.find("h1")
    assert (h1.text, "page-title" in h1.classes) == (heading, True)
    assert texts(page, "title") == [f"{heading} · {SITE_NAME}"]


def test_error_page_is_a_panel_with_a_home_button(client):
    page = Page(body_text(client.get("/does-not-exist")))

    [panel] = page.find(class_="error-panel")
    assert "panel" in panel.classes
    [home] = panel.find("a", "btn-primary", href="/")
    assert home.text == "Ana sayfaya dön"


def test_skip_link_is_the_first_link_and_targets_main(pages):
    for name, body in pages.items():
        assert Page(body).all("a")[0]["href"] == "#main-content", name


def test_favicon_is_linked_on_every_page_and_served(client, pages):
    for name, body in pages.items():
        [icon] = Page(body).all("link", rel="icon")
        assert icon["type"] == "image/svg+xml", name
        assert urlsplit(icon["href"]).path == "/static/favicon.svg", name

    response = client.get(icon["href"])

    assert response.status_code == 200
    assert response.mimetype == "image/svg+xml"
    assert response.get_data(as_text=True).lstrip().startswith("<svg")


def test_home_screen_icon_is_linked_and_named_movierec(client, pages):
    for name, body in pages.items():
        [touch] = Page(body).all("link", rel="apple-touch-icon")
        assert urlsplit(touch["href"]).path == "/static/apple-touch-icon.png", name
        [title] = Page(body).all("meta", name="apple-mobile-web-app-title")
        assert title["content"] == SITE_NAME, name

    response = client.get(touch["href"])
    png = response.get_data()

    assert response.status_code == 200
    assert response.mimetype == "image/png"
    # The size iOS asks for, read from the PNG header (IHDR width and height).
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert (int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")) == (180, 180)


def test_favicon_is_an_ink_square_with_the_red_dot():
    root = ElementTree.parse(FAVICON).getroot()
    tokens = _tokens()
    _, _, width, height = (float(n) for n in root.get("viewBox").split())
    shapes = [element for element in root.iter() if element.tag != f"{SVG}svg"]

    # Two shapes and nothing else: the old film strip had nine rectangles.
    assert [shape.tag for shape in shapes] == [f"{SVG}rect", f"{SVG}circle"]
    rect, circle = shapes
    assert rect.get("fill").upper() == tokens["--color-ink"].upper()
    assert float(rect.get("width", 0)) == width and float(rect.get("height", 0)) == height
    assert float(rect.get("x", 0)) == 0 and float(rect.get("y", 0)) == 0
    # A rounded square: a corner radius of about 18% of the side.
    assert 0.16 <= float(rect.get("rx")) / width <= 0.2
    assert circle.get("fill").upper() == tokens["--color-accent"].upper()
    assert (float(circle.get("cx")), float(circle.get("cy"))) == (width / 2, height / 2)
    # Big enough to read as a dot at 16px: at least 6 of the 16 pixels across.
    assert 2 * float(circle.get("r")) / width >= 6 / 16


def _weights_by_family():
    """{token: font weights} of the stylesheets: rules that set var(--font-display) count
    for the display face, every other font-weight for the text face."""
    weights = {"--font-sans": {"400"}, "--font-display": set()}
    for sheet in (STATIC / "css").glob("*.css"):
        for _, selector, body in _css_rules(sheet.read_text(encoding="utf-8")):
            family = "--font-display" if "var(--font-display)" in body else "--font-sans"
            for value in re.findall(r"font-weight:\s*([^;\s]+)", body):
                weights[family].add({"normal": "400", "bold": "700"}.get(value, value))
            if family == "--font-display":
                # The display face is never left at the inherited (text) weight.
                assert "font-weight:" in body, (sheet.name, selector)
    return weights


def _font_faces():
    """The descriptors of every @font-face rule in base.css, each as {name: value}."""
    css = re.sub(r"/\*.*?\*/", "", BASE_CSS.read_text(encoding="utf-8"), flags=re.S)
    return [
        dict(re.findall(r"([\w-]+):\s*([^;]+);", body))
        for body in re.findall(r"@font-face\s*\{(.*?)\}", css, flags=re.S)
    ]


def _in_unicode_range(char, unicode_range):
    """True when char is in a unicode-range value such as "U+0000-00FF, U+0131"."""
    for part in unicode_range.split(","):
        low, _, high = part.strip().removeprefix("U+").partition("-")
        if int(low, 16) <= ord(char) <= int(high or low, 16):
            return True
    return False


def test_the_two_plex_families_come_in_the_weights_the_css_uses():
    faces = _font_faces()
    weights = {}
    for face in faces:
        weights.setdefault(face["font-family"], set()).add(face["font-weight"])

    # IBM Plex Sans is one variable font for its weights; the condensed face has a file each.
    assert weights == {
        '"IBM Plex Sans"': {"400 600"},
        '"IBM Plex Sans Condensed"': {"500", "600", "700"},
    }
    # A weight the CSS uses but no face holds would be faked by the browser; an unused one is
    # waste. Body text is 400 without a font-weight rule.
    used = _weights_by_family()
    assert used["--font-sans"] == {"400", "500", "600"}
    assert used["--font-display"] == weights['"IBM Plex Sans Condensed"']
    assert {face["font-display"] for face in faces} == {"swap"}


def test_the_fonts_are_woff2_files_of_this_site_with_their_licence():
    fonts = STATIC / "fonts"
    files = set()
    for face in _font_faces():
        [name] = re.findall(r'^url\("\.\./fonts/([\w-]+\.woff2)"\) format\("woff2"\)$', face["src"])
        assert (fonts / name).read_bytes()[:4] == b"wOF2", name
        files.add(name)

    assert files == {path.name for path in fonts.glob("*.woff2")}
    licence = (fonts / "OFL.txt").read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE Version 1.1" in licence


@pytest.mark.parametrize("char", "çğıöşüÇĞİÖŞÜ")
def test_every_font_weight_has_the_turkish_letters(char):
    ranges = {}
    for face in _font_faces():
        key = (face["font-family"], face["font-weight"])
        ranges.setdefault(key, []).append(face["unicode-range"])

    assert ranges
    for key, unicode_ranges in ranges.items():
        assert any(_in_unicode_range(char, value) for value in unicode_ranges), key


def test_headings_and_bold_text_use_a_loaded_weight():
    # Browsers draw h1-h3 and strong at 700, which is not loaded and would be faked.
    for selector in ("h1", "h2", "h3", "strong", "b"):
        assert "font-weight: 600" in _declarations(BASE_CSS, selector), selector


def _static_urls(body):
    page = Page(body)
    urls = [tag["href"] for tag in page.all("link") if tag.get("href", "").startswith("/static/")]
    urls += [tag["src"] for tag in page.all("script") if tag.get("src", "").startswith("/static/")]
    urls += [tag["src"] for tag in page.all("img") if tag.get("src", "").startswith("/static/")]
    return urls


def test_static_urls_carry_a_stable_content_version(client, pages):
    for name, body in pages.items():
        urls = _static_urls(body)
        stylesheets = [urlsplit(url).path for url in urls if urlsplit(url).path.endswith(".css")]
        expected = ["css/base.css", "css/cards.css", *PAGE_STYLESHEETS[name]]
        assert stylesheets == [f"/static/{sheet}" for sheet in expected], name
        for url in urls:
            assert re.fullmatch(r"[0-9a-f]{12}", parse_qs(urlsplit(url).query).get("v", [""])[0]), (
                name,
                url,
            )
    assert any("/static/js/movie_detail.js?" in url for url in _static_urls(pages["movie"]))
    assert any("/static/js/my_lists.js?" in url for url in _static_urls(pages["my_lists"]))
    assert _static_urls(body_text(client.get("/"))) == _static_urls(pages["home"])


def test_versioned_static_files_are_cached_for_a_year(client):
    [stylesheet] = [
        url for url in _static_urls(body_text(client.get("/"))) if "css/base.css" in url
    ]

    response = client.get(stylesheet)

    assert response.status_code == 200
    assert response.cache_control.max_age == 31536000
    assert response.cache_control.public


def test_unversioned_static_urls_are_revalidated(client):
    # Module imports such as "./list_store.js" cannot carry asset_url()'s ?v=, so a year-long
    # cache would keep an old module next to a new page after a deploy.
    response = client.get("/static/js/list_store.js")

    assert response.status_code == 200
    assert response.cache_control.max_age == 0


def test_asset_version_follows_the_file_content(tmp_path):
    (tmp_path / "a.css").write_text("body{}")
    (tmp_path / "js").mkdir()
    (tmp_path / "js" / "b.js").write_text("let x;")
    before = build_asset_versions(tmp_path)
    (tmp_path / "a.css").write_text("body{color:red}")
    after = build_asset_versions(tmp_path)

    assert set(before) == {"a.css", "js/b.js"}
    assert before["a.css"] != after["a.css"]
    assert before["js/b.js"] == after["js/b.js"]


# --- Icons: a local SVG sprite with the Font Awesome Free paths, not the icon font ---
def _sprite_symbols():
    """{id: viewBox} of every <symbol> in static/icons.svg."""
    root = ElementTree.parse(SPRITE).getroot()
    return {symbol.get("id"): symbol.get("viewBox") for symbol in root.iter(f"{SVG}symbol")}


def _svgs(markup):
    """Every inline svg element in the markup (HTML or an element)."""
    return (Page(markup) if isinstance(markup, str) else markup).find("svg")


def _icons(markup):
    """(svg attributes, use attributes, title texts) of every sprite icon in the markup."""
    icons = []
    for svg in _svgs(markup):
        if svg.classes[:1] != ["icon"]:
            continue
        [use] = svg.find("use")
        icons.append((svg.attrs, use.attrs, texts(svg, "title")))
    return icons


def test_every_inline_svg_is_a_sprite_icon(every_page):
    # The logo is HTML text, so every svg on a page is an icon of the sprite.
    for name, body in every_page.items():
        for svg in _svgs(body):
            assert svg.classes[:1] == ["icon"], (name, svg)


@pytest.fixture
def every_page(pages, client):
    """Every page type, plus the searches without results and with a warning, and a flash."""
    with client.session_transaction() as session:
        session["_flashes"] = [("warning", FLASH_TEXT)]
    flash = body_text(client.get("/movie/155"))
    return {
        **pages,
        "empty": body_text(client.get(SEARCHES["empty"])),
        "warning": body_text(client.get(SEARCHES["warning"])),
        "flash": flash,
    }


def test_no_page_or_asset_uses_font_awesome(every_page):
    for name, body in every_page.items():
        assert "font-awesome" not in body.lower(), name
        classes = {c for _, tag in Page(body).tags for c in tag.get("class", "").split()}
        assert {c for c in classes if c in {"fa", "fas", "far"} or c.startswith("fa-")} == set()
    scripts = list((STATIC / "js").glob("*.js"))
    assert scripts
    for path in [*(STATIC / "css").glob("*.css"), *scripts]:
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"font-family:[^;]*Font Awesome", source), path.name
        assert not re.search(r"content:\s*[\"']\\f", source), path.name
        assert not re.search(r"\bfa-[a-z]", source), path.name


def test_every_icon_draws_a_symbol_of_the_versioned_sprite(every_page):
    symbols = _sprite_symbols()
    for name, body in every_page.items():
        assert _icons(body), name
        for svg, use, _ in _icons(body):
            url = urlsplit(use["href"])
            assert url.path == "/static/icons.svg", (name, use)
            assert re.fullmatch(r"[0-9a-f]{12}", parse_qs(url.query).get("v", [""])[0]), name
            assert url.fragment in symbols, (name, use)
            # The symbol's viewBox gives the icon its Font Awesome width at a height of 1em.
            assert svg["viewbox"] == symbols[url.fragment], (name, use)
            assert svg["class"].split()[0] == "icon", (name, svg)


def test_icons_are_hidden_from_screen_readers_on_every_page(every_page):
    for name, body in every_page.items():
        for svg, _, titles in _icons(body):
            assert svg.get("aria-hidden") == "true", (name, svg)
            assert svg.get("focusable") == "false", (name, svg)
            assert "role" not in svg and titles == [], (name, svg)


def _render_icon(app, call):
    with app.test_request_context("/"):
        return render_template_string('{% from "_icon.html" import icon %}' + call)


def test_icon_macro_draws_a_decorative_icon_by_default(app_module):
    [(svg, use, titles)] = _icons(_render_icon(app_module.app, "{{ icon('film') }}"))

    assert svg["class"] == "icon"
    assert svg["aria-hidden"] == "true"
    assert svg["focusable"] == "false"
    assert "role" not in svg and titles == []
    assert use["href"].endswith("#film")


def test_icon_macro_with_a_label_is_announced_as_an_image(app_module):
    markup = _render_icon(app_module.app, "{{ icon('star', label='Puan <b>8</b>') }}")

    [(svg, use, titles)] = _icons(markup)
    assert svg["role"] == "img"
    assert "aria-hidden" not in svg
    # The label is text: escaped, so it adds no element.
    assert titles == ["Puan <b>8</b>"]
    assert not Page(markup).find("b")
    assert use["href"].endswith("#star")


def test_icon_macro_takes_extra_classes_and_refuses_unknown_names(app_module):
    markup = _render_icon(app_module.app, "{{ icon('spinner', class='icon-spin') }}")

    [(svg, _, _)] = _icons(markup)
    assert svg["class"] == "icon icon-spin"
    # A typo fails the render instead of drawing an empty box.
    with pytest.raises(KeyError):
        _render_icon(app_module.app, "{{ icon('no-such-icon') }}")


def _icons_in_use():
    """Icon names the templates draw with icon('...') and the scripts swap in ({icon: '...'}
    and {activeIcon: '...'} in movie_detail.js)."""
    templates = " ".join(path.read_text(encoding="utf-8") for path in TEMPLATES.glob("*.html"))
    script_paths = list((STATIC / "js").glob("*.js"))
    assert script_paths
    scripts = " ".join(path.read_text(encoding="utf-8") for path in script_paths)
    return set(re.findall(r"\bicon\('([a-z-]+)'", templates)) | set(
        re.findall(r"\b(?:icon|activeIcon):\s*'([a-z-]+)'", scripts)
    )


def test_sprite_holds_only_the_icons_in_use_with_the_font_awesome_licence():
    source = SPRITE.read_text(encoding="utf-8")
    symbols = _sprite_symbols()

    assert FONT_AWESOME_LICENCE in source
    assert "Icons: CC BY 4.0" in source
    # Font Awesome icons are 512 units high; only the width varies.
    assert all(re.fullmatch(r"0 0 \d+ 512", view_box) for view_box in symbols.values())
    # Every symbol is drawn somewhere, and every icon drawn is in the sprite.
    assert set(symbols) == _icons_in_use()


def test_list_buttons_have_an_outline_and_a_solid_icon_of_the_same_width():
    # movie_detail.js swaps the outline (regular) icon for the solid one when a film is in a
    # list; the svg keeps its viewBox, so both shapes must share it.
    symbols = _sprite_symbols()

    for name in ("heart", "bookmark"):
        assert symbols[f"{name}-regular"] == symbols[name], name


def test_the_regular_icons_are_font_awesomes_own_paths():
    # Downloaded from @fortawesome/fontawesome-free@6.5.1/svgs/regular/, not redrawn.
    paths = {
        symbol.get("id"): symbol.find(f"{SVG}path").get("d")
        for symbol in ElementTree.parse(SPRITE).getroot().iter(f"{SVG}symbol")
    }

    assert paths["heart-regular"].startswith("M225.8 468.2l-2.5-2.3L48.1 303.2C17.4 274.7")
    assert paths["heart-regular"].endswith("s-12.4-2.5-16.9-7z")
    assert paths["bookmark-regular"] == (
        "M0 48C0 21.5 21.5 0 48 0l0 48V441.4l130.1-92.9c8.3-6 19.6-6 27.9 0L336 441.4V48H48V0"
        "H336c26.5 0 48 21.5 48 48V488c0 9-5 17.2-13 21.3s-17.6 3.4-24.9-1.8L192 397.5 37.9 "
        "507.5c-7.3 5.2-16.9 5.9-24.9 1.8S0 497 0 488V48z"
    )


def test_sprite_is_served_as_svg_with_the_long_cache(client, pages):
    [sprite] = {use["href"].split("#")[0] for _, use, _ in _icons(pages["home"])}

    response = client.get(sprite)

    assert response.status_code == 200
    assert response.mimetype == "image/svg+xml"
    assert response.cache_control.max_age == 31536000


def test_every_page_loads_the_shared_script_once(pages):
    for name, body in pages.items():
        scripts = [
            tag
            for tag in Page(body).all("script")
            if tag.get("src", "").startswith("/static/js/base.js")
        ]
        assert len(scripts) == 1, name


def test_templates_have_no_inline_styles_scripts_or_handlers():
    for template in TEMPLATES.glob("*.html"):
        source = template.read_text(encoding="utf-8")
        assert "<style" not in source, template.name
        assert not re.search(r"<script(?![^>]*\bsrc=)", source), template.name
        assert " onclick=" not in source, template.name
        assert ' style="' not in source, template.name


def _flash_areas(body):
    return [tag for _, tag in Page(body).tags if "flash-messages" in tag.get("class", "").split()]


@pytest.mark.parametrize("path", ["/", "/movie/155", "/actor/Christian+Bale", "/my_lists", "/nope"])
def test_flashed_message_is_shown_once_on_every_page(person_client, path):
    with person_client.session_transaction() as session:
        session["_flashes"] = [("warning", FLASH_TEXT)]

    body = body_text(person_client.get(path))

    assert body.count(FLASH_TEXT) == 1
    [area] = Page(body).find(class_="flash-messages")
    [alert] = area.find(class_="alert")
    assert FLASH_TEXT in alert.text
    assert alert.find("button", data_dismiss="alert")


def test_flash_area_is_left_out_without_messages(pages):
    for name, body in pages.items():
        assert _flash_areas(body) == [], name


def test_header_search_is_on_every_page(pages):
    for name, body in pages.items():
        assert 'name="search_query"' in body, name


def test_every_page_links_to_my_lists_and_marks_it_current_there(pages):
    for name, body in pages.items():
        [link] = Page(body).all("a", href="/my_lists")
        assert (link.get("aria-current") == "page") == (name == "my_lists"), name
        # Phones show only the icon; the text stays the link's name for screen readers.
        assert texts(body, "a", href="/my_lists") == ["Listelerim"], name


def _brand_link(body):
    """The header's home link: its attributes and its element."""
    [link] = Page(body).find("a", "header-brand")
    return link.attrs, link


def _logos(markup):
    """Every logo element in `markup` (HTML or an element)."""
    return (Page(markup) if isinstance(markup, str) else markup).find("span", "logo")


def test_header_brand_is_a_home_link_named_movierec_around_the_logo(pages):
    for name, body in pages.items():
        link, element = _brand_link(body)

        assert link["href"] == "/", name
        assert link["aria-label"] == SITE_NAME, name
        [logo] = _logos(element)
        # The wordmark is drawn, not read: the link's label names it.
        assert logo.attrs == {"class": "logo", "aria-hidden": "true"}, name
        # The film icon of the old brand is gone, and the logo is no image.
        assert not element.find("svg") and not element.find("img"), name


def test_home_page_h1_is_the_brand_and_says_movierec(pages):
    # The brand is the home page's only h1; its text is for screen readers, the logo for eyes.
    _, link = _brand_link(pages["home"])
    [h1] = link.find("h1")

    assert texts(h1, "span", "sr-only") == [SITE_NAME]
    [logo] = _logos(h1)
    assert h1.text == f"{logo.text} {SITE_NAME}"
    for name in ("movie", "my_lists"):
        assert not _brand_link(pages[name])[1].find("h1"), name


def test_theme_script_runs_in_the_head_before_the_stylesheets(pages):
    # A blocking classic script ahead of the stylesheets puts the stored theme on <html>
    # before the first paint, so a stored dark choice never flashes light.
    for name, body in pages.items():
        head = body[: body.index("</head>")]
        # The first script of the page; a page's own scripts may follow in the head.
        script = Page(head).all("script")[0]
        assert urlsplit(script["src"]).path == "/static/js/theme.js", name
        assert not {"async", "defer", "type"} & set(script), name
        assert head.index("/static/js/theme.js") < head.index('rel="stylesheet"'), name
        [meta] = Page(head).all("meta", name="color-scheme")
        assert meta["content"] == "light dark", name


def test_every_header_has_the_theme_button(pages):
    for name, body in pages.items():
        [header] = Page(body).find("header", "header")
        [markup] = [
            button for button in header.find("button") if "data-theme-toggle" in button.attrs
        ]
        button = markup.attrs

        assert button["type"] == "button", name
        # The server cannot know the theme; static/js/theme.js sets the label before it is used.
        assert button["aria-label"] == button["title"] == "Koyu temaya geç", name
        # The moon shows on the light page, the sun on the dark one (css/base.css).
        assert [urlsplit(use["href"]).fragment for _, use, _ in _icons(markup)] == ["moon", "sun"]
    # Without the script the button could not work, so it is not shown.
    assert "display: none" in _declarations(BASE_CSS, ":root:not([data-theme]) .theme-toggle")


def _render_logo(app, call="{{ logo() }}"):
    with app.test_request_context("/"):
        return render_template_string('{% from "_logo.html" import logo %}' + call)


def test_logo_is_a_red_dot_before_movie_rec(app_module):
    markup = _render_logo(app_module.app)
    root = ElementTree.fromstring(markup)
    dot, rec = list(root)

    assert (root.tag, root.get("class"), root.get("aria-hidden")) == ("span", "logo", "true")
    assert "".join(root.itertext()) == SITE_NAME
    # An empty box the CSS paints as the dot, right before the word: no space widens the gap.
    assert (dot.tag, dot.get("class"), dot.text, dot.tail) == ("span", "logo-dot", None, "Movie")
    # "Movie" and "Rec" differ in weight, so the camel-cased name never reads as "movi erec".
    assert (rec.tag, rec.get("class"), rec.text, rec.tail) == ("b", "logo-rec", "Rec", None)


def test_logo_is_sized_in_em_and_dotted_in_the_accent():
    # No measured pixel numbers: the logo scales with the text around it and keeps its shape
    # in a fallback face.
    for selector in (".logo", ".logo-dot", ".logo-rec"):
        rule = _declarations(BASE_CSS, selector)
        assert rule and not re.search(r"\d(px|rem)\b", rule), selector
    assert "background-color: var(--color-accent)" in _declarations(BASE_CSS, ".logo-dot")


def test_footer_signs_with_the_logo_in_ink(client):
    [footer] = Page(body_text(client.get("/"))).find("footer")
    [brand] = footer.find("p", "footer-brand")

    assert _logos(brand)
    assert texts(brand, "span", "sr-only") == [SITE_NAME]
    assert "color: var(--color-ink)" in _declarations(BASE_CSS, ".footer-brand")


def test_nothing_user_facing_keeps_the_old_site_name():
    sources = [
        *TEMPLATES.glob("*.html"),
        *(path for path in STATIC.rglob("*") if path.suffix in {".css", ".js", ".svg"}),
        *(REPO / "movierec").rglob("*.py"),
        *(REPO / "tests").rglob("*.py"),
        *(REPO / "tests" / "js").glob("*.mjs"),
        REPO / "README.md",
    ]

    for path in sources:
        assert not OLD_SITE_NAME.search(path.read_text(encoding="utf-8")), path.name


@pytest.mark.parametrize("sheet", STYLESHEETS)
def test_stylesheets_use_only_the_mobile_first_breakpoints(sheet):
    css = (STATIC / "css" / sheet).read_text(encoding="utf-8")
    widths = re.findall(r"\((min|max)-width:\s*([^)]+)\)", css)

    assert [kind for kind, _ in widths if kind == "max"] == [], sheet
    assert {width.strip() for _, width in widths} <= set(BREAKPOINTS), sheet


def test_focus_is_visible_and_motion_can_be_reduced():
    css = BASE_CSS.read_text(encoding="utf-8")

    assert "var(--color-accent)" in _declarations(BASE_CSS, ":focus-visible")
    # The accent would vanish on the black header.
    assert "outline-color:" in _declarations(BASE_CSS, ".header :focus-visible")
    assert "@media (prefers-reduced-motion: reduce)" in css


@pytest.mark.parametrize("sheet", STYLESHEETS)
def test_stylesheets_draw_no_shadows_gradients_pills_or_hover_lifts(sheet):
    # The editorial look separates with space and hairlines, not depth effects.
    rules = list(_css_rules((STATIC / "css" / sheet).read_text(encoding="utf-8")))

    for _, selector, body in rules:
        for shadow in re.findall(r"box-shadow:\s*([^;]+)", body):
            assert shadow.strip() == "none", (sheet, selector)
        assert "gradient(" not in body, (sheet, selector)
        # Radii are the 6px/8px tokens; only the round play button is 50%.
        for radius in re.findall(r"border-radius:\s*([^;]+)", body):
            for size in re.findall(r"([\d.]+)(px|rem|em)", radius):
                assert float(size[0]) * (16 if size[1] != "px" else 1) <= 8, (sheet, selector)
        if ":hover" in selector:
            assert "transform" not in body, (sheet, selector)


def test_every_css_variable_in_use_is_defined():
    sheets = [path.read_text(encoding="utf-8") for path in (STATIC / "css").glob("*.css")]
    defined = {name for css in sheets for name in re.findall(r"(--[\w-]+)\s*:", css)}
    used = {name for css in sheets for name in re.findall(r"var\(\s*(--[\w-]+)", css)}

    assert used - defined == set()
