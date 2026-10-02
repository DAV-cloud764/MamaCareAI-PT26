"""
Stage 5: Storage (PDF 3.5).

Three distinct concerns, deliberately kept separate rather than collapsed into
one table:

  - Object storage  -> raw files (already written during ingestion)
  - Relational DB   -> structured state, versions, review assignments
  - Search index    -> full-text search over the translated Swahili text

WHY NOT ONE TABLE
They have genuinely different shapes, sizes, and access patterns: blobs are
huge and immutable, state is small and updated constantly, the index is a
rebuildable read model. Collapsing them means every query fights the wrong
storage engine, and the "just one more column" table becomes unqueryable
within a year.

WHAT THIS STAGE ACTUALLY DOES
By the time we get here the content is already persisted. This stage's job is
to make it FINDABLE and to open a review task — it is the handoff from machine
processing to human judgement.
"""

from __future__ import annotations

import uuid

from ..domain.enums import ResourceStatus, VersionAuthorKind
from ..domain.models import ContentVersion, Resource, TranslationUnit
from ..observability.metrics import Metrics
from ..ports.job_queue import JobQueue
from ..ports.repositories import (
    DocumentRepository,
    ResourceRepository,
    ReviewRepository,
    VersionRepository,
)
from ..ports.search_index import SearchIndex
from ..services.review_service import ReviewService
from .base import Stage, StageResult


class StoreStage(Stage):
    """Opens a human review task for a translated or native-Swahili resource."""

    def __init__(
        self,
        *,
        resources: ResourceRepository,
        queue: JobQueue,
        reviews: ReviewRepository,
        documents: DocumentRepository,
        versions: VersionRepository,
        search: SearchIndex,
        review_service: ReviewService,
        max_attempts: int = 5,
        metrics: Metrics | None = None,
    ) -> None:
        super().__init__(
            resources=resources, queue=queue, reviews=reviews, max_attempts=max_attempts, metrics=metrics
        )
        self._documents = documents
        self._versions = versions
        self._search = search
        self._review_service = review_service

    @property
    def name(self) -> str:
        return "store"

    @property
    def accepts(self) -> frozenset[ResourceStatus]:
        # TRANSLATED = normal path. LANGUAGE_DETECTED = the already-Swahili
        # shortcut from stage 3, which has no machine translation to index.
        return frozenset({ResourceStatus.TRANSLATED, ResourceStatus.LANGUAGE_DETECTED})

    def handle(self, resource: Resource) -> StageResult:
        """Open a reviewable version and hand the document to a human.

        Nothing is indexed here. This stage used to write the translation into
        the search index, and `ReviewService` re-indexed on every edit, so
        unapproved content reached the production index twice before any human
        had looked at it. `PublishStage` is now the only writer to that index,
        which is the only way "nothing unapproved is published" is true rather
        than merely intended. The review queue reads from the repository
        (`list_by_status`), so removing this costs the reviewers nothing.

        Two paths, and both must produce a version:

          - a resource that went through translation already has one; the
            reviewer's job is to check and correct the machine output;
          - a native-Swahili source skipped translation entirely, so there is
            nothing to compare against and nothing for the review UI to align.
            It gets a version built from the extracted blocks, with the source
            text as its own translation. The reviewer sees the same side-by-side
            layout as every other document instead of an empty pane, and their
            edits diff against the original the same way a machine version does.
        """
        version = self._versions.get_latest(resource.resource_id)
        if version is None:
            version = self._source_version(resource)

        self._review_service.enqueue_for_review(resource, version)

        return StageResult(
            next_status=ResourceStatus.STORED,
            next_stage="review",
        )

    def _source_version(self, resource: Resource) -> ContentVersion:
        """Build version 1 for content that is already Swahili.

        Without this, a native-Swahili resource has no version at all: the old
        code built a flattened string inline and indexed it, so nothing
        reviewable was ever persisted. A reviewer opening it got an empty right
        pane and no way to record what they checked.
        """
        document = self._documents.get_document(resource.resource_id)
        return ContentVersion(
            version_id=str(uuid.uuid4()),
            resource_id=resource.resource_id,
            version_number=1,
            author_kind=VersionAuthorKind.MACHINE,
            author_id=None,
            units=tuple(
                TranslationUnit(
                    order=block.order,
                    source_text=block.text,
                    translated_text=block.text,
                    kind=block.kind,
                    confidence=None,
                )
                for block in document.blocks
            ),
            engine="source:already-target-language",
            note="Source was already Swahili; no machine translation was needed.",
        )
