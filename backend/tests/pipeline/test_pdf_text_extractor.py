from __future__ import annotations

import unicodedata
from io import BytesIO

import pymupdf
import pytest

from modules.pipeline.adapters.extractors.pdf_text_extractor import PdfTextExtractor
from modules.pipeline.domain.errors import ExtractionError


def make_pdf(
    pages: list[list[str]],
    *,
    title: str | None = None,
    author: str | None = None,
    creation_date: str | None = None,
    encrypted: bool = False,
) -> bytes:
    document = pymupdf.open()
    if title or author or creation_date:
        document.set_metadata(
            {
                "title": title or "",
                "author": author or "",
                "creationDate": creation_date or "",
            }
        )
    for page_lines in pages:
        page = document.new_page()
        for line_number, text in enumerate(page_lines):
            page.insert_text((72, 72 + line_number * 18), text)

    output = BytesIO()
    save_kwargs = {
        "encryption": pymupdf.PDF_ENCRYPT_AES_256,
        "owner_pw": "owner-password",
        "user_pw": "user-password",
    } if encrypted else {}
    document.save(output, **save_kwargs)
    document.close()
    return output.getvalue()


def test_can_handle_rejects_non_pdf_bytes() -> None:
    assert not PdfTextExtractor().can_handle("application/pdf", b"not a PDF")


def test_can_handle_rejects_corrupt_pdf() -> None:
    assert not PdfTextExtractor().can_handle("application/pdf", b"%PDF-corrupt")


def test_can_handle_rejects_insufficient_embedded_text() -> None:
    content = make_pdf([["short"]])
    assert not PdfTextExtractor(min_chars_per_page=100).can_handle(
        "application/pdf", content
    )


def test_can_handle_accepts_text_pdf_above_threshold() -> None:
    text = " ".join(["Maternal health guidance"] * 20)
    content = make_pdf([[text]])
    assert PdfTextExtractor(min_chars_per_page=100).can_handle(
        "application/pdf", content
    )


def test_extract_returns_ordered_normalized_blocks_and_metadata() -> None:
    content = make_pdf(
        [
            [
                "Maternal health guidance",
                "Cafe\u0301 and pregnancy care information.",
            ],
        ],
        title="Maternal Guide",
        author="Ministry of Health",
        creation_date="D:20260102112233Z",
    )

    document = PdfTextExtractor().extract(
        "resource-1",
        content,
        metadata={"source": "test"},
    )

    assert document.resource_id == "resource-1"
    assert document.blocks
    assert [block.order for block in document.blocks] == list(range(len(document.blocks)))
    assert all(
        unicodedata.is_normalized("NFC", block.text) for block in document.blocks
    )
    assert document.title == "Maternal Guide"
    assert document.author == "Ministry of Health"
    assert document.published_date is not None
    assert document.published_date.year == 2026
    assert document.source_metadata["source"] == "test"
    assert document.source_metadata["page_count"] == 1
    assert document.source_metadata["title"] == "Maternal Guide"
    assert document.source_metadata["author"] == "Ministry of Health"


def test_extract_raises_when_no_usable_text_is_produced() -> None:
    content = make_pdf([[]])

    with pytest.raises(ExtractionError):
        PdfTextExtractor().extract("empty-resource", content, metadata={})


def test_encrypted_pdf_is_declined_and_rejected() -> None:
    content = make_pdf([["Confidential maternal guidance"]], encrypted=True)
    extractor = PdfTextExtractor(min_chars_per_page=1)

    assert not extractor.can_handle("application/pdf", content)
    with pytest.raises(ExtractionError):
        extractor.extract("encrypted-resource", content, metadata={})
