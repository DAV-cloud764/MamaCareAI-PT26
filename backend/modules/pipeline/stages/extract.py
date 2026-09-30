"""
Stage 2: Extraction & normalization (PDF 3.2).

Converts raw bytes into the ONE common schema — `NormalizedDocument` — that
every later stage speaks.

THIS IS THE MOST IMPORTANT ARCHITECTURAL BOUNDARY IN THE PIPELINE.
Everything before it is source-type-specific. Everything after it is not.
Translation, review, and publication never learn whether the content started
as a PDF, a web page, or a video transcript. Protect that boundary: if
source-type knowledge leaks past this stage, every later stage grows branches
and the pipeline stops being extensible.

IMPLEMENTATION NOTES:
- This implements the handle() method for the extract stage
- Converts raw bytes into NormalizedDocument
- Requires Dev B to complete: registries (PIPE-08), object store (PIPE-07)
"""

from __future__ import annotations

from backend.modules.pipeline.domain.enums import ResourceStatus
from backend.modules.pipeline.domain.errors import ExtractionError
from backend.modules.pipeline.domain.models import (
    Resource,
)
from backend.modules.pipeline.observability.metrics import Metrics
from backend.modules.pipeline.ports.job_queue import JobQueue
from backend.modules.pipeline.ports.object_store import ObjectStore
from backend.modules.pipeline.ports.repositories import (
    DocumentRepository,
    ResourceRepository,
    ReviewRepository,
)
from backend.modules.pipeline.registry import ExtractorRegistry

from .base import Stage, StageResult


class ExtractStage(Stage):
    """Produces a NormalizedDocument from stored raw bytes."""

    def __init__(
        self,
        *,
        resources: ResourceRepository,
        queue: JobQueue,
        reviews: ReviewRepository,
        documents: DocumentRepository,
        extractors: ExtractorRegistry,
        object_store: ObjectStore,
        max_attempts: int = 5,
        metrics: Metrics | None = None,
    ) -> None:
        super().__init__(
            resources=resources, queue=queue, reviews=reviews, max_attempts=max_attempts, metrics=metrics
        )
        self._documents = documents
        self._extractors = extractors
        self._object_store = object_store

    @property
    def name(self) -> str:
        return "extract"

    @property
    def accepts(self) -> frozenset[ResourceStatus]:
        return frozenset({ResourceStatus.FETCHED})

    def handle(self, resource: Resource) -> StageResult:
        """Load raw bytes, run the right extractor, save the normalized document.

        IMPLEMENTATION following the TODO order:

        1. LOAD the raw bytes
        2. SELECT THE EXTRACTOR VIA THE REGISTRY, with fallback
        3. EXTRACT to NormalizedDocument
        4. VALIDATE THE OUTPUT
        5. PERSIST and return StageResult
        """
        # 1. LOAD the raw bytes
        if not resource.raw_object_key:
            raise ExtractionError("No raw_object_key set on resource", resource_id=resource.resource_id)

        try:
            content = self._object_store.get(resource.raw_object_key)
        except KeyError as exc:
            # Missing key is a permanent error - do not retry forever
            raise ExtractionError(f"Raw content not found in object store: {exc}", resource_id=resource.resource_id)

        # 2. SELECT THE EXTRACTOR VIA THE REGISTRY, with fallback
        # The registry tries candidates in priority order and picks the first whose can_handle returns True
        try:
            extractor = self._extractors.select("application/octet-stream", content)
        except ExtractionError as exc:
            # No extractor could handle this content
            raise ExtractionError(f"No extractor available for content: {exc}", resource_id=resource.resource_id)

        # 3. EXTRACT
        try:
            document = extractor.extract(
                resource.resource_id,
                content,
                metadata=resource.source_metadata,
            )
        except (ExtractionError, ValueError, TypeError) as exc:
            # Catch specific extraction-related errors
            raise ExtractionError(f"Extraction failed: {exc}", resource_id=resource.resource_id)

        # 4. VALIDATE THE OUTPUT before accepting it
        # - at least one TextBlock
        if not document.blocks:
            raise ExtractionError("Extractor returned document with zero blocks", resource_id=resource.resource_id)

        # - total text length above a sane minimum
        total_chars = sum(len(block.text) for block in document.blocks)
        min_chars = 12  # Configurable, but a reasonable minimum
        if total_chars < min_chars:
            raise ExtractionError(
                f"Extracted text too short ({total_chars} chars, minimum {min_chars})",
                resource_id=resource.resource_id,
            )

        # - `order` values are unique and contiguous
        orders = [block.order for block in document.blocks]
        if len(set(orders)) != len(orders):
            raise ExtractionError("TextBlock order values are not unique", resource_id=resource.resource_id)

        if orders != list(range(len(orders))):
            raise ExtractionError(
                "TextBlock order values are not contiguous and do not start at 0",
                resource_id=resource.resource_id,
            )

        # 5. PERSIST and continue
        # IDEMPOTENCY: save_document overwrites by resource_id, so re-running is safe by construction
        self._documents.save_document(document)

        # Merge metadata
        merged_metadata = {**resource.source_metadata, **document.source_metadata}

        return StageResult(
            next_status=ResourceStatus.EXTRACTED,
            next_stage="detect_language",
            resource_changes={"source_metadata": merged_metadata},
            details={
                "blocks": len(document.blocks),
                "total_chars": total_chars,
                "extractor": type(extractor).__name__,
            },
        )
