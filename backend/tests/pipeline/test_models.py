"""
Domain model tests.

`NormalizedDocument.raw_text` is the flattened projection that language
detection and search indexing read. It is the one derived property in the
domain, and it is what tells us whether `blocks` really is the single source
of truth: if this test ever needs a second copy of the text on the document,
`blocks` has stopped being authoritative.
"""

from __future__ import annotations

from datetime import UTC, datetime

from modules.pipeline.domain.models import NormalizedDocument, TextBlock


def make_document(*blocks: TextBlock) -> NormalizedDocument:
    return NormalizedDocument(
        resource_id="r1",
        title="Ujauzito na matibabu",
        author=None,
        published_date=datetime(2026, 5, 10, tzinfo=UTC),
        blocks=blocks,
    )


def test_raw_text_of_no_blocks_is_empty() -> None:
    assert make_document().raw_text == ""


def test_raw_text_of_one_block_has_no_separator() -> None:
    document = make_document(TextBlock(order=0, kind="heading", text="Dalili za hatari"))

    assert document.raw_text == "Dalili za hatari"


def test_raw_text_joins_blocks_with_a_blank_line_in_order() -> None:
    document = make_document(
        TextBlock(order=0, kind="heading", text="Dalili za hatari"),
        TextBlock(order=1, kind="paragraph", text="Nenda kituo cha afya mara moja."),
        TextBlock(order=2, kind="list_item", text=" kutoka kwa damu nyingi"),
    )

    assert document.raw_text == (
        "Dalili za hatari\n\nNenda kituo cha afya mara moja.\n\n kutoka kwa damu nyingi"
    )


def test_raw_text_ignores_block_kind() -> None:
    """Structure is carried by `blocks`; `raw_text` is prose only.

    A detector must not see the word "heading" in the text it classifies.
    """
    document = make_document(
        TextBlock(order=0, kind="heading", text="Dalili za hatari"),
        TextBlock(order=1, kind="list_item", text="Kutoka kwa damu nyingi"),
    )

    assert "heading" not in document.raw_text
    assert "list_item" not in document.raw_text
