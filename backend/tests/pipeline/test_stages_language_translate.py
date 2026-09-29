from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from modules.pipeline.adapters.translation.chunker import Chunk, Chunker
from modules.pipeline.domain.enums import ResourceStatus, SourceType, VersionAuthorKind
from modules.pipeline.domain.errors import TranslationError
from modules.pipeline.domain.models import ContentVersion, Resource, TextBlock
from modules.pipeline.ports.job_queue import JobQueue
from modules.pipeline.ports.language_detector import DetectionResult, LanguageDetector
from modules.pipeline.ports.repositories import (
    DocumentRepository,
    ResourceRepository,
    ReviewRepository,
    VersionRepository,
)
from modules.pipeline.ports.translator import TranslatedChunk, Translator
from modules.pipeline.stages.detect_language import DetectLanguageStage
from modules.pipeline.stages.translate import TranslateStage


@dataclass
class Document:
    raw_text: str
    blocks: tuple[TextBlock, ...]


class FakeDocuments:
    def __init__(self, document: Document) -> None:
        self.document = document

    def get_document(self, resource_id: str) -> Document:
        return self.document


class FakeDetector:
    def __init__(self, result: DetectionResult) -> None:
        self.result = result
        self.calls: list[str] = []

    def detect(self, text: str) -> DetectionResult:
        self.calls.append(text)
        return self.result


class FakeVersions:
    def __init__(self, machine_version: ContentVersion | None = None) -> None:
        self.machine_version = machine_version
        self.saved: list[ContentVersion] = []

    def get_machine_version(self, resource_id: str) -> ContentVersion | None:
        return self.machine_version

    def save_version(self, version: ContentVersion) -> None:
        self.saved.append(version)
        self.machine_version = version


class FakeTranslator:
    engine_name = "fake-translator-v1"

    def __init__(
        self, *, supported: bool = True, results: list[TranslatedChunk] | None = None
    ) -> None:
        self.supported = supported
        self.results = results
        self.calls: list[list[str]] = []

    def supports(self, source_language: str, target_language: str) -> bool:
        return self.supported

    def translate_batch(
        self, texts: list[str], *, source_language: str, target_language: str
    ) -> list[TranslatedChunk]:
        self.calls.append(texts)
        if self.results is not None:
            return self.results
        return [TranslatedChunk(f"[sw] {text}", 0.8) for text in texts]


class FakeChunker:
    def chunk(self, blocks: tuple[TextBlock, ...]) -> list[Chunk]:
        return [Chunk(block.text, (block.order,)) for block in blocks]

    def reassemble(
        self,
        blocks: tuple[TextBlock, ...],
        chunks: list[Chunk],
        translations: list[str],
    ) -> list[tuple[int, str]]:
        return [
            (chunk.block_orders[0], translation)
            for chunk, translation in zip(chunks, translations, strict=True)
        ]


def make_resource(
    *, status: ResourceStatus = ResourceStatus.EXTRACTED, **changes: object
) -> Resource:
    fields: dict[str, Any] = {
        "resource_id": "resource-1",
        "source_type": SourceType.WEB,
        "source_url": "https://example.org/article",
        "status": status,
        "detected_language": None,
        "language_confidence": None,
        "source_metadata": {},
    }
    fields.update(changes)
    return Resource(**fields)


def make_document() -> Document:
    return Document(
        raw_text="Maternal health guidance.",
        blocks=(
            TextBlock(order=0, kind="heading", text="Antenatal care"),
            TextBlock(order=1, kind="paragraph", text="Attend every appointment."),
        ),
    )


def build_detect_stage(detector: FakeDetector) -> DetectLanguageStage:
    return DetectLanguageStage(
        resources=cast(ResourceRepository, object()),
        queue=cast(JobQueue, object()),
        reviews=cast(ReviewRepository, object()),
        documents=cast(DocumentRepository, FakeDocuments(make_document())),
        detector=cast(LanguageDetector, detector),
    )


def build_translate_stage(
    versions: FakeVersions, translator: FakeTranslator
) -> TranslateStage:
    return TranslateStage(
        resources=cast(ResourceRepository, object()),
        queue=cast(JobQueue, object()),
        reviews=cast(ReviewRepository, object()),
        documents=cast(DocumentRepository, FakeDocuments(make_document())),
        versions=cast(VersionRepository, versions),
        translator=cast(Translator, translator),
        chunker=cast(Chunker, FakeChunker()),
    )


def test_low_confidence_routes_to_human_confirmation() -> None:
    alternatives = (("en", 0.45), ("fr", 0.31))
    detector = FakeDetector(DetectionResult("en", 0.45, alternatives))

    result = build_detect_stage(detector).handle(make_resource())

    assert detector.calls == ["Maternal health guidance."]
    assert result.next_status is ResourceStatus.NEEDS_LANGUAGE_CONFIRMATION
    assert result.next_stage is None
    assert result.resource_changes == {
        "detected_language": "en",
        "language_confidence": 0.45,
    }
    assert result.details["alternatives"] == alternatives


def test_already_swahili_skips_translation() -> None:
    result = build_detect_stage(FakeDetector(DetectionResult("sw", 0.95))).handle(
        make_resource()
    )

    assert result.next_status is ResourceStatus.LANGUAGE_DETECTED
    assert result.next_stage == "store"


def test_other_language_routes_to_translation() -> None:
    result = build_detect_stage(FakeDetector(DetectionResult("en", 0.95))).handle(
        make_resource()
    )

    assert result.next_status is ResourceStatus.LANGUAGE_DETECTED
    assert result.next_stage == "translate"


def test_human_confirmed_language_is_not_overwritten() -> None:
    detector = FakeDetector(DetectionResult("en", 0.99))
    result = build_detect_stage(detector).handle(
        make_resource(
            status=ResourceStatus.NEEDS_LANGUAGE_CONFIRMATION,
            detected_language="sw",
            language_confidence=1.0,
            source_metadata={"language_confirmed_by": "reviewer-1"},
        )
    )

    assert detector.calls == []
    assert result.next_stage == "store"
    assert result.resource_changes["detected_language"] == "sw"


def test_creates_machine_version_one_with_engine_recorded() -> None:
    versions = FakeVersions()
    translator = FakeTranslator()
    result = build_translate_stage(versions, translator).handle(
        make_resource(status=ResourceStatus.LANGUAGE_DETECTED, detected_language="en")
    )

    version = versions.saved[0]
    assert result.next_status is ResourceStatus.TRANSLATED
    assert result.next_stage == "store"
    assert version.version_number == 1
    assert version.author_kind is VersionAuthorKind.MACHINE
    assert version.engine == "fake-translator-v1"


def test_rerun_does_not_create_a_second_machine_version() -> None:
    versions = FakeVersions()
    translator = FakeTranslator()
    stage = build_translate_stage(versions, translator)
    resource = make_resource(
        status=ResourceStatus.LANGUAGE_DETECTED, detected_language="en"
    )

    stage.handle(resource)
    second_result = stage.handle(resource)

    assert len(versions.saved) == 1
    assert len(translator.calls) == 1
    assert second_result.details["reused_version"] == versions.saved[0].version_id


def test_length_mismatch_from_translator_raises() -> None:
    versions = FakeVersions()
    translator = FakeTranslator(results=[])

    with pytest.raises(TranslationError, match="different number of chunks"):
        build_translate_stage(versions, translator).handle(
            make_resource(
                status=ResourceStatus.LANGUAGE_DETECTED, detected_language="en"
            )
        )

    assert versions.saved == []


def test_block_structure_survives_translation() -> None:
    versions = FakeVersions()
    stage = build_translate_stage(versions, FakeTranslator())
    stage.handle(
        make_resource(status=ResourceStatus.LANGUAGE_DETECTED, detected_language="en")
    )

    units = versions.saved[0].units
    assert [
        (unit.order, unit.kind, unit.source_text, unit.translated_text)
        for unit in units
    ] == [
        (0, "heading", "Antenatal care", "[sw] Antenatal care"),
        (
            1,
            "paragraph",
            "Attend every appointment.",
            "[sw] Attend every appointment.",
        ),
    ]
