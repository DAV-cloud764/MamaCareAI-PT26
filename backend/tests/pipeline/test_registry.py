from __future__ import annotations

import pytest
from backend.modules.pipeline.domain.enums import SourceType
from backend.modules.pipeline.domain.errors import (
    ExtractionError,
    UnsupportedSourceType,
)
from backend.modules.pipeline.domain.models import NormalizedDocument
from backend.modules.pipeline.ports.extractor import ContentExtractor
from backend.modules.pipeline.ports.fetcher import FetchResult, SourceFetcher
from backend.modules.pipeline.registry import ExtractorRegistry, FetcherRegistry


class FakeFetcher(SourceFetcher):
    """Fake fetcher for testing registry without I/O."""

    def __init__(self, handled_source_type: SourceType) -> None:
        self.handled_source_type = handled_source_type

    @property
    def source_type(self) -> SourceType:
        return self.handled_source_type

    def fetch(self, source_url: str) -> FetchResult:
        return FetchResult(content=source_url.encode(), content_type="text/plain")


class FakeExtractor(ContentExtractor):
    """Fake extractor for testing registry without I/O."""

    def __init__(self, name: str, handles: bool) -> None:
        self.name = name
        self.handles = handles

    def can_handle(self, content_type: str, content: bytes) -> bool:
        return self.handles

    def extract(
        self,
        resource_id: str,
        content: bytes,
        *,
        metadata: dict[str, object],
    ) -> NormalizedDocument:
        raise AssertionError("extract should not be called by registry selection tests")


def test_fetcher_registry_returns_registered_fetcher():
    """Registry returns the exact fetcher instance registered."""
    registry = FetcherRegistry()
    web_fetcher = FakeFetcher(SourceType.WEB)
    pdf_fetcher = FakeFetcher(SourceType.PDF)

    registry.register(web_fetcher)
    registry.register(pdf_fetcher)

    assert registry.get(SourceType.WEB) is web_fetcher
    assert registry.get(SourceType.PDF) is pdf_fetcher


def test_fetcher_registry_rejects_duplicate_source_type():
    """Registering the same source type twice raises ValueError."""
    registry = FetcherRegistry()
    registry.register(FakeFetcher(SourceType.WEB))

    with pytest.raises(ValueError, match="web"):
        registry.register(FakeFetcher(SourceType.WEB))


def test_fetcher_registry_raises_for_unsupported_source_type():
    """Getting an unregistered source type raises UnsupportedSourceType."""
    registry = FetcherRegistry()

    with pytest.raises(UnsupportedSourceType, match="video"):
        registry.get(SourceType.VIDEO)


def test_extractor_registry_selects_highest_priority_match():
    """When both extractors can handle it, highest priority wins."""
    registry = ExtractorRegistry()
    low_priority = FakeExtractor(name="low", handles=True)
    high_priority = FakeExtractor(name="high", handles=True)

    registry.register(low_priority, priority=50)
    registry.register(high_priority, priority=100)

    assert registry.select("application/pdf", b"%PDF") is high_priority


def test_extractor_registry_falls_back_when_high_priority_declines():
    """When high priority says no, registry tries the next priority."""
    registry = ExtractorRegistry()
    text_layer = FakeExtractor(name="text", handles=False)
    ocr = FakeExtractor(name="ocr", handles=True)

    registry.register(ocr, priority=50)
    registry.register(text_layer, priority=100)

    assert registry.select("application/pdf", b"%PDF") is ocr


def test_extractor_registry_raises_when_no_extractor_can_handle_payload():
    """When no extractor can handle it, raise ExtractionError with content_type."""
    registry = ExtractorRegistry()
    registry.register(FakeExtractor(name="text", handles=False), priority=100)
    registry.register(FakeExtractor(name="ocr", handles=False), priority=50)

    with pytest.raises(ExtractionError, match="application/pdf"):
        registry.select("application/pdf", b"%PDF")