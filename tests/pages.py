"""Helpers for parsing rendered pages in the template tests.

Page parses the markup into elements, so tests can ask for an element by tag, class or
attribute and read its text, whatever the attribute order or the whitespace in the template.
"""

from html.parser import HTMLParser

# Elements without an end tag; everything else is closed by its end tag.
_VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}


class Element:
    """One element of a parsed page: its tag, attributes, children (elements and text)."""

    def __init__(self, tag, attrs, parent=None):
        self.tag = tag
        self.attrs = attrs
        self.parent = parent
        self.children = []

    def __repr__(self):
        return f"<{self.tag} {self.attrs}>"

    @property
    def classes(self):
        return (self.attrs.get("class") or "").split()

    @property
    def text(self):
        """The text inside, with each tag read as a space and whitespace collapsed."""
        return " ".join(self._raw_text().split())

    def _raw_text(self):
        return "".join(
            child if isinstance(child, str) else f" {child._raw_text()} " for child in self.children
        )

    def descendants(self):
        for child in self.children:
            if isinstance(child, Element):
                yield child
                yield from child.descendants()

    def find(self, tag=None, class_=None, **attrs):
        """Every element inside with this tag, this class among its classes and these
        attribute values (underscores in names stand for dashes), in document order."""
        return _matching(self.descendants(), tag, class_, attrs)


def _matching(elements, tag, class_, attrs):
    wanted = {key.rstrip("_").replace("_", "-"): value for key, value in attrs.items()}
    return [
        element
        for element in elements
        if (tag is None or element.tag == tag)
        and (class_ is None or class_ in element.classes)
        and all(element.attrs.get(key) == value for key, value in wanted.items())
    ]


class Page(HTMLParser):
    """A rendered page: its start tags in order (tags) and its element tree (root)."""

    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.root = Element("#document", {})
        self._open = [self.root]
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        element = Element(tag, dict(attrs), parent=self._open[-1])
        self.tags.append((tag, element.attrs))
        self._open[-1].children.append(element)
        if tag not in _VOID:
            self._open.append(element)

    def handle_startendtag(self, tag, attrs):
        element = Element(tag, dict(attrs), parent=self._open[-1])
        self.tags.append((tag, element.attrs))
        self._open[-1].children.append(element)

    def handle_endtag(self, tag):
        # Close the innermost open element with this tag; a stray end tag changes nothing.
        for depth in range(len(self._open) - 1, 0, -1):
            if self._open[depth].tag == tag:
                del self._open[depth:]
                return

    def handle_data(self, data):
        self._open[-1].children.append(data)

    def all(self, tag, **attrs):
        """Return the attribute dicts of every `tag` whose attributes include `attrs`."""
        wanted = {key.replace("_", "-"): value for key, value in attrs.items()}
        return [
            found
            for name, found in self.tags
            if name == tag and all(found.get(key) == value for key, value in wanted.items())
        ]

    def with_id(self, element_id):
        """Return the attribute dicts of every start tag whose id is `element_id`."""
        return [found for _, found in self.tags if found.get("id") == element_id]

    def find(self, tag=None, class_=None, **attrs):
        """Every element with this tag, class and attribute values; see Element.find."""
        return self.root.find(tag, class_, **attrs)

    @property
    def text(self):
        return self.root.text


def body_text(response):
    """The decoded body of a test client response."""
    return response.get_data(as_text=True)


def texts(markup, tag=None, class_=None, **attrs):
    """The text of every matching element in `markup`: an Element, a Page or HTML."""
    if isinstance(markup, str):
        markup = Page(markup)
    return [element.text for element in markup.find(tag, class_, **attrs)]
