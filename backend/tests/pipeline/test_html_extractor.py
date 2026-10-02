"""
HtmlExtractor tests (PIPE-18).

No network, no real fetcher — raw HTML fixtures passed straight to the
extractor. Covers the three things the port contract cares about:
  1. structure is preserved (headings / paragraphs / lists come back as
     separate TextBlocks, in document order),
  2. boilerplate (nav / site header / footer) is dropped,
  3. an empty extraction raises ExtractionError instead of returning a
     document with zero blocks.
"""

from __future__ import annotations

import pytest

from modules.pipeline.adapters.extractors.html_extractor import HtmlExtractor
from modules.pipeline.domain.errors import ExtractionError


def make_extractor() -> HtmlExtractor:
    return HtmlExtractor()


ARTICLE = b"""<!doctype html>
<html lang="en">
<head>
  <title>Pregnancy Danger Signs</title>
  <meta name="author" content="Ministry of Health">
  <meta property="article:published_time" content="2026-05-10">
</head>
<body>
  <nav>
    <a href="/">Home</a>
    <a href="/about">About us</a>
  </nav>
  <header><h1>Site Header Banner</h1></header>
  <div id="content">
    <h1>Pregnancy Danger Signs</h1>
    <p>Go to the nearest health facility immediately if you experience any of the
    following danger signs during pregnancy. This sentence is intentionally long
    so that the readability extractor treats the page as a real article and not
    as a tiny snippet of boilerplate.</p>
    <h2>You need urgent care if</h2>
    <ul>
      <li>you have severe bleeding</li>
      <li>you have fits or convulsions</li>
      <li>you have a severe headache that will not go away</li>
    </ul>
    <p>The final paragraph summarises the guidance and reminds every reader to
    save the emergency telephone number somewhere visible.</p>
  </div>
  <footer>Copyright 2026 Ministry of Health</footer>
</body>
</html>
"""

TABLE_PAGE = b"""<!doctype html>
<html>
<body>
  <div id="content">
    <h1>Danger signs at a glance</h1>
    <p>This table lists the danger signs and the moment you should seek care.
    The surrounding text is deliberately padded out with enough words so the
    extractor keeps the page as one article and does not drop it entirely.</p>
    <table>
      <tr><th>Symptom</th><th>When to seek care</th></tr>
      <tr><td>Fever</td><td>Any time of day</td></tr>
      <tr><td>Bleeding</td><td>Immediately</td></tr>
    </table>
    <p>If in doubt, always go to the facility - it is better to be checked and
    find nothing wrong than to wait at home and regret it later.</p>
  </div>
</body>
</html>
"""


def test_can_handle_by_content_type() -> None:
    assert make_extractor().can_handle("text/html; charset=utf-8", b"irrelevant")


def test_can_handle_by_sniffing_when_header_is_wrong() -> None:
    extractor = make_extractor()
    # Servers sometimes serve HTML as application/octet-stream — sniff, don't trust.
    assert extractor.can_handle(
        "application/octet-stream", b"<html><body>hi</body></html>"
    )


def test_can_handle_rejects_non_html() -> None:
    assert not make_extractor().can_handle("application/pdf", b"%PDF-1.4")


def test_extract_preserves_structure_and_order() -> None:
    doc = make_extractor().extract(
        "r1", ARTICLE, metadata={"content_type": "text/html; charset=utf-8"}
    )

    assert [(b.kind, b.order) for b in doc.blocks] == [
        ("heading", 0),
        ("paragraph", 1),
        ("heading", 2),
        ("list_item", 3),
        ("list_item", 4),
        ("list_item", 5),
        ("paragraph", 6),
    ]
    assert doc.blocks[0].text == "Pregnancy Danger Signs"
    assert doc.blocks[3].text == "you have severe bleeding"


def test_extract_drops_nav_header_and_footer() -> None:
    doc = make_extractor().extract(
        "r1", ARTICLE, metadata={"content_type": "text/html; charset=utf-8"}
    )

    body_text = " ".join(b.text for b in doc.blocks).lower()
    assert "home" not in body_text
    assert "about us" not in body_text
    assert "site header banner" not in body_text
    assert "copyright 2026" not in body_text


def test_extract_serializes_table_rows_and_flags_tables() -> None:
    doc = make_extractor().extract(
        "r1", TABLE_PAGE, metadata={"content_type": "text/html; charset=utf-8"}
    )

    rows = [b for b in doc.blocks if b.kind == "list_item"]
    assert rows == [doc.blocks[i] for i in (1, 2, 3)] or any(
        "Symptom" in b.text for b in rows
    )
    assert any(b.text == "Fever | Any time of day" for b in rows)
    assert doc.source_metadata.get("contains_tables") is True


def test_extract_reads_title_and_date_metadata() -> None:
    doc = make_extractor().extract(
        "r1", ARTICLE, metadata={"content_type": "text/html; charset=utf-8"}
    )

    assert doc.title == "Pregnancy Danger Signs"
    assert doc.published_date is not None
    assert doc.published_date.year == 2026
    assert doc.published_date.month == 5


@pytest.mark.parametrize(
    "empty_html",
    [
        b"<html><body></body></html>",
        b"<!doctype html><html><head><title>t</title></head><body><nav>x</nav></body></html>",
        b"   ",
    ],
)
def test_extract_raises_on_empty_or_boilerplate_only_input(empty_html: bytes) -> None:
    with pytest.raises(ExtractionError):
        make_extractor().extract(
            "r1", empty_html, metadata={"content_type": "text/html"}
        )


def test_extract_normalizes_nbsp_and_zero_width() -> None:
    html = (
        '<!doctype html><html><body><div id="content"><h1>Title</h1>'
        "<p>A&nbsp;sentence\u200bwith zero\u200bwidth.</p>"
        "<p>Some more text to make the article long enough for the extractor "
        "to keep it as a real article rather than dropping it entirely. "
        "Padding padding padding padding padding padding padding.</p>"
        "</div></body></html>"
    ).encode("utf-8")
    doc = make_extractor().extract("r1", html, metadata={"content_type": "text/html"})

    paragraph = next(b.text for b in doc.blocks if b.kind == "paragraph")
    assert "\u200b" not in paragraph
    assert "  " not in paragraph
