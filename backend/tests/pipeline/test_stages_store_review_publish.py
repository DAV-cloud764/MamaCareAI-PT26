"""
Tests for the store, review, and publish stages (PIPE-11).

**Owner: Dev C** (see the Sprint 1 split in `docs/PIPELINE_BACKLOG.md`).

Fakes live in this file for now — `tests/pipeline/fakes.py` is Dev A's PIPE-06
file and is still a template. When it lands, these tests should switch to
`build_test_container()` without changing what they assert.

What is covered, per the plan:
  - StoreStage indexes BOTH the "has an MT version" path and the
    "already-Swahili, no version" path (the easy-to-miss one).
  - StoreStage opens exactly one review assignment.
  - ReviewStage always returns next_stage=None (the pipeline parks here).
  - PublishStage blocks on a failed compliance check and never publishes.
  - PublishStage publishes the LATEST version, not version 1, when a human
    edit exists — the worst-bug guard.
  - PublishStage records who approved and which version in its details.
"""

from __future__ import annotations

import pytest

from modules.pipeline.domain.enums import ResourceStatus, SourceType, VersionAuthorKind
from modules.pipeline.domain.errors import InvalidStateTransition, TransientError
from modules.pipeline.domain.models import (
    AuditEvent,
    ContentVersion,
    NormalizedDocument,
    Resource,
    ReviewAssignment,
    TextBlock,
    TranslationUnit,
)
from modules.pipeline.ports.search_index import IndexedResource
from modules.pipeline.stages.publish import PublishStage
from modules.pipeline.stages.review import ReviewStage
from modules.pipeline.stages.store import StoreStage

# --- fakes (see file docstring) ----------------------------------------------


class FakeVersionRepository:
    def __init__(self, versions: dict[str, list[ContentVersion]] | None = None) -> None:
        self._by_resource: dict[str, list[ContentVersion]] = versions or {}

    def save_version(self, version: ContentVersion) -> None:
        self._by_resource.setdefault(version.resource_id, []).append(version)

    def get_latest(self, resource_id: str) -> ContentVersion | None:
        versions = self._by_resource.get(resource_id, [])
        return max(versions, key=lambda v: v.version_number) if versions else None

    def list_versions(self, resource_id: str) -> list[ContentVersion]:
        """Needed to resolve a pinned `approved_version_id` back to a version."""
        return sorted(
            self._by_resource.get(resource_id, []), key=lambda v: v.version_number
        )


class FakeDocumentRepository:
    def __init__(self) -> None:
        self._documents: dict[str, NormalizedDocument] = {}

    def save_document(self, document: NormalizedDocument) -> None:
        self._documents[document.resource_id] = document

    def get_document(self, resource_id: str) -> NormalizedDocument:
        return self._documents[resource_id]


class FakeSearchIndex:
    def __init__(self) -> None:
        self.indexed: dict[str, IndexedResource] = {}

    def index(self, resource: IndexedResource) -> None:
        self.indexed[resource.resource_id] = resource

    def search(self, query: str, *, limit: int = 20, offset: int = 0) -> list:
        return []


class FakeReviewService:
    def __init__(self) -> None:
        self.assignments: dict[str, ReviewAssignment] = {}
        self.calls: list[tuple[Resource, ContentVersion | None]] = []

    def enqueue_for_review(
        self, resource: Resource, version: ContentVersion | None
    ) -> ReviewAssignment:
        self.calls.append((resource, version))
        existing = self.assignments.get(resource.resource_id)
        if existing is not None:
            return existing
        assignment = ReviewAssignment(
            assignment_id=f"a-{resource.resource_id}",
            resource_id=resource.resource_id,
            reviewer_id=None,  # unclaimed
        )
        self.assignments[resource.resource_id] = assignment
        return assignment


class FakeComplianceGate:
    def __init__(self, *, allowed: bool, reason: str | None = None) -> None:
        self._allowed = allowed
        self._reason = reason

    def evaluate(self, resource: Resource):
        from modules.pipeline.services.compliance import ComplianceDecision

        return ComplianceDecision(allowed=self._allowed, reason=self._reason)


class FakeKnowledgeHandoff:
    def __init__(self) -> None:
        self.handoffs: list[dict] = []

    def handoff_published_content(
        self,
        resource_id: str,
        source_url: str,
        title: str | None,
        translated_text: str,
        version_number: int,
        language: str,
        metadata: dict,
    ) -> None:
        self.handoffs.append(
            {
                "resource_id": resource_id,
                "source_url": source_url,
                "title": title,
                "translated_text": translated_text,
                "version_number": version_number,
                "language": language,
                "metadata": metadata,
            }
        )


class FakeResourceRepository:
    def get(self, resource_id: str) -> Resource:
        raise NotImplementedError


class FakeJobQueue:
    def publish(self, job) -> None:
        self.published = getattr(self, "published", [])
        self.published.append(job)


class FakeReviewRepository:
    def __init__(self) -> None:
        self.audit: list[AuditEvent] = []

    def append_audit(self, event: AuditEvent) -> None:
        self.audit.append(event)


# --- builders ----------------------------------------------------------------


def make_resource(*, status: ResourceStatus, **overrides) -> Resource:
    fields = {
        "resource_id": "r1",
        "source_type": SourceType.WEB,
        "source_url": "https://example.org/article",
        "status": status,
        "source_metadata": {},
    }
    fields.update(overrides)
    return Resource(**fields)


def make_version(
    *, resource_id: str = "r1", version_number: int = 1, text: str = "translated"
) -> ContentVersion:
    return ContentVersion(
        version_id=f"v{resource_id}-{version_number}",
        resource_id=resource_id,
        version_number=version_number,
        author_kind=VersionAuthorKind.HUMAN
        if version_number > 1
        else VersionAuthorKind.MACHINE,
        author_id=None if version_number == 1 else "reviewer-7",
        units=(TranslationUnit(order=0, source_text="source", translated_text=text),),
    )


def make_document(*, resource_id: str = "r1") -> NormalizedDocument:
    return NormalizedDocument(
        resource_id=resource_id,
        title="Already-Swahili title",
        author=None,
        published_date=None,
        blocks=(
            TextBlock(order=0, kind="paragraph", text="Habari za afya ya mama."),
            TextBlock(
                order=1, kind="paragraph", text="Tafadhali wasiliana na daktari."
            ),
        ),
    )


def build_store_stage(
    **overrides,
) -> tuple[StoreStage, FakeSearchIndex, FakeReviewService]:
    search = FakeSearchIndex()
    review_service = FakeReviewService()
    versions = overrides.get("versions") or FakeVersionRepository()
    documents = overrides.get("documents") or FakeDocumentRepository()
    stage = StoreStage(
        resources=FakeResourceRepository(),
        queue=FakeJobQueue(),
        reviews=FakeReviewRepository(),
        documents=documents,
        versions=versions,
        search=search,
        review_service=review_service,
    )
    return stage, search, review_service


def build_publish_stage(**overrides) -> tuple[PublishStage, FakeSearchIndex]:
    search = FakeSearchIndex()
    compliance = overrides.get("compliance") or FakeComplianceGate(allowed=True)
    versions = overrides.get("versions") or FakeVersionRepository()
    knowledge = overrides.get("knowledge")
    stage = PublishStage(
        resources=FakeResourceRepository(),
        queue=FakeJobQueue(),
        reviews=FakeReviewRepository(),
        versions=versions,
        search=search,
        compliance_gate=compliance,
        knowledge_handoff=knowledge,
    )
    return stage, search


def approved_resource(*, version_number: int = 1, **overrides) -> Resource:
    """An APPROVED resource whose approval is pinned to a specific version.

    `PublishStage` publishes the version approval named, not the newest one.
    Every publish test therefore has to say which version was approved — which
    is the point: a test that forgot to pin one would otherwise be quietly
    asserting that "whatever is newest" is good enough to publish.
    """
    metadata = {
        "approved_by": "reviewer-7",
        "approved_version_id": f"vr1-{version_number}",
        "approved_version_number": version_number,
    }
    metadata.update(overrides.pop("source_metadata", None) or {})
    return make_resource(
        status=ResourceStatus.APPROVED, source_metadata=metadata, **overrides
    )


# --- store stage -------------------------------------------------------------


def test_store_does_not_index_and_opens_one_assignment() -> None:
    """Store must not write to the index.

    It used to, and `ReviewService` re-indexed on every edit, so unapproved
    content reached the production index twice before a human had looked at it.
    `PublishStage` is the only writer now.
    """
    versions = FakeVersionRepository()
    versions.save_version(make_version(resource_id="r1", version_number=1))
    stage, search, review_service = build_store_stage(versions=versions)

    result = stage.handle(
        make_resource(status=ResourceStatus.TRANSLATED, source_metadata={"title": "T"})
    )

    assert result.next_status == ResourceStatus.STORED
    assert result.next_stage == "review"
    assert search.indexed == {}
    assert len(review_service.calls) == 1
    assert len(review_service.assignments) == 1
    # The existing machine version is what the reviewer is asked to check.
    _resource, handed_version = review_service.calls[0]
    assert handed_version is not None
    assert handed_version.version_number == 1


def test_store_builds_a_reviewable_version_for_native_swahili() -> None:
    """A source that is already Swahili still needs a version to review.

    There was no machine translation, so the old code built a flattened string
    inline and indexed it. Nothing reviewable was persisted, so a reviewer
    opened the document to an empty right pane with no way to record what they
    had checked.
    """
    documents = FakeDocumentRepository()
    documents.save_document(make_document())
    versions = FakeVersionRepository()  # deliberately empty: no MT version
    stage, search, review_service = build_store_stage(
        versions=versions, documents=documents
    )

    result = stage.handle(make_resource(status=ResourceStatus.LANGUAGE_DETECTED))

    assert result.next_status == ResourceStatus.STORED
    assert search.indexed == {}

    _resource, version = review_service.calls[0]
    assert version is not None
    assert version.version_number == 1
    assert version.author_kind is VersionAuthorKind.MACHINE
    assert version.engine == "source:already-target-language"
    # Aligned with the source blocks by `order`, so the review UI can render
    # both panes the same way it does for translated content.
    assert [unit.order for unit in version.units] == [0, 1]
    assert version.units[0].source_text == "Habari za afya ya mama."
    assert version.units[0].translated_text == "Habari za afya ya mama."
    assert version.units[0].kind == "paragraph"


def test_store_handle_running_twice_does_not_double_the_review_assignments() -> None:
    versions = FakeVersionRepository()
    versions.save_version(make_version())
    stage, _search, review_service = build_store_stage(versions=versions)
    resource = make_resource(status=ResourceStatus.TRANSLATED)

    stage.handle(resource)
    stage.handle(resource)  # at-least-once delivery: the job can run twice

    assert len(review_service.assignments) == 1


# --- review stage ------------------------------------------------------------


def test_review_stage_moves_to_in_review_and_stops() -> None:
    stage = ReviewStage(
        resources=FakeResourceRepository(),
        queue=FakeJobQueue(),
        reviews=FakeReviewRepository(),
    )

    result = stage.handle(make_resource(status=ResourceStatus.STORED))

    assert result.next_status == ResourceStatus.IN_REVIEW
    assert result.next_stage is None  # a human, not a worker, drives it from here


# --- publish stage -----------------------------------------------------------


def test_publish_stage_blocks_on_compliance_failure_and_never_publishes() -> None:
    compliance = FakeComplianceGate(allowed=False, reason="unknown licence")
    versions = FakeVersionRepository()
    versions.save_version(make_version())
    stage, search = build_publish_stage(compliance=compliance, versions=versions)

    result = stage.handle(approved_resource(version_number=1))

    assert result.next_status == ResourceStatus.BLOCKED_LICENSING
    assert result.next_stage is None
    assert result.details["reason"] == "unknown licence"
    assert "r1" not in search.indexed


def test_publish_stage_publishes_the_latest_version_not_version_one() -> None:
    versions = FakeVersionRepository()
    versions.save_version(make_version(version_number=1, text="machine"))
    versions.save_version(make_version(version_number=2, text="human-edited"))
    stage, search = build_publish_stage(versions=versions)

    result = stage.handle(approved_resource(version_number=2))

    assert result.next_status == ResourceStatus.PUBLISHED
    indexed = search.indexed["r1"]
    assert indexed.version_number == 2
    assert indexed.translated_text == "human-edited"
    assert indexed.status == "published"


def test_publish_stage_records_who_approved_and_which_version() -> None:
    versions = FakeVersionRepository()
    versions.save_version(make_version(version_number=2, text="approved"))
    stage, _search = build_publish_stage(versions=versions)

    result = stage.handle(approved_resource(version_number=2))

    assert result.details["approved_by"] == "reviewer-7"
    assert result.details["approved_version"] == 2


def test_publish_stage_publishes_the_version_approval_named_not_the_newest() -> None:
    """Approval pins a version; publishing "whatever is newest" is not that.

    Version 1 was approved. Version 2 was then appended by something else — a
    stray edit, a retried job, a second tab. Publishing `get_latest()` would put
    content in the production index that no human ever approved, while the
    review UI still showed an approved-looking document.
    """
    versions = FakeVersionRepository()
    versions.save_version(make_version(version_number=1, text="the approved one"))
    versions.save_version(make_version(version_number=2, text="never approved"))
    stage, search = build_publish_stage(versions=versions)

    result = stage.handle(approved_resource(version_number=1))

    assert result.next_status == ResourceStatus.PUBLISHED
    indexed = search.indexed["r1"]
    assert indexed.version_number == 1
    assert indexed.translated_text == "the approved one"


def test_publish_stage_refuses_a_resource_with_no_pinned_approval() -> None:
    """No approved version recorded means nothing is authorised to publish.

    This used to fall back to the newest version, which is how an unapproved
    document could reach the index. It now refuses instead.
    """
    versions = FakeVersionRepository()
    versions.save_version(make_version(version_number=1, text="unreviewed"))
    stage, search = build_publish_stage(versions=versions)

    with pytest.raises(InvalidStateTransition, match="no.*approved version"):
        stage.handle(make_resource(status=ResourceStatus.APPROVED))

    assert search.indexed == {}


def test_publish_stage_records_an_audit_event_for_the_transition() -> None:
    """Every action writes an audit event; publication was not doing that.

    `ComplianceGate` documents that every evaluation, pass or fail, must be
    recorded by the calling stage. Publication recorded neither its own
    transition nor its own gate result.
    """
    reviews = FakeReviewRepository()
    versions = FakeVersionRepository()
    versions.save_version(make_version(version_number=1, text="approved"))
    stage = PublishStage(
        resources=FakeResourceRepository(),
        queue=FakeJobQueue(),
        reviews=reviews,
        versions=versions,
        search=FakeSearchIndex(),
        compliance_gate=FakeComplianceGate(allowed=True),
    )

    stage.handle(approved_resource(version_number=1))

    assert [event.action for event in reviews.audit] == ["publish"]
    event = reviews.audit[0]
    assert event.actor_id == "system:publish"
    assert event.from_status is ResourceStatus.APPROVED
    assert event.to_status is ResourceStatus.PUBLISHED
    assert event.details["approved_version"] == 1


def test_a_blocked_publication_is_audited_too() -> None:
    """A refusal is exactly the kind of decision a later reader needs."""
    reviews = FakeReviewRepository()
    versions = FakeVersionRepository()
    versions.save_version(make_version(version_number=1, text="blocked"))
    stage = PublishStage(
        resources=FakeResourceRepository(),
        queue=FakeJobQueue(),
        reviews=reviews,
        versions=versions,
        search=FakeSearchIndex(),
        compliance_gate=FakeComplianceGate(allowed=False, reason="unknown licence"),
    )

    stage.handle(approved_resource(version_number=1))

    assert [event.action for event in reviews.audit] == ["publish_blocked"]
    assert reviews.audit[0].to_status is ResourceStatus.BLOCKED_LICENSING
    assert reviews.audit[0].details["reason"] == "unknown licence"


def test_a_failed_handoff_publishes_nothing_and_is_retriable() -> None:
    """The handoff runs BEFORE the index write, and failure is retriable.

    It used to run after, while its own comment said it should not fail the
    publish — so a handoff error raised after the index was already mutated,
    leaving the resource in APPROVED with a PUBLISHED index entry. Two
    subsystems disagreeing about one document, with nothing to reconcile them.
    """
    class BrokenHandoff:
        def handoff_published_content(self, **kwargs) -> None:
            raise RuntimeError("knowledge module unavailable")

    reviews = FakeReviewRepository()
    versions = FakeVersionRepository()
    versions.save_version(make_version(version_number=1, text="never indexed"))
    search = FakeSearchIndex()
    stage = PublishStage(
        resources=FakeResourceRepository(),
        queue=FakeJobQueue(),
        reviews=reviews,
        versions=versions,
        search=search,
        compliance_gate=FakeComplianceGate(allowed=True),
        knowledge_handoff=BrokenHandoff(),
    )

    with pytest.raises(TransientError):
        stage.handle(approved_resource(version_number=1))

    assert search.indexed == {}
    assert reviews.audit == []


# --- knowledge handoff tests (PIPE-32) --------------------------------------


def test_knowledge_handoff_called_when_configured() -> None:
    """Test that knowledge handoff is called when configured."""
    knowledge = FakeKnowledgeHandoff()
    versions = FakeVersionRepository()
    versions.save_version(make_version(version_number=2, text="swahili text"))
    stage, _search = build_publish_stage(versions=versions, knowledge=knowledge)

    result = stage.handle(
        approved_resource(
            version_number=2,
            source_metadata={"title": "Test Title"},
            detected_language="sw",
        )
    )

    assert result.next_status == ResourceStatus.PUBLISHED
    assert len(knowledge.handoffs) == 1
    assert knowledge.handoffs[0]["resource_id"] == "r1"
    assert knowledge.handoffs[0]["translated_text"] == "swahili text"
    assert knowledge.handoffs[0]["version_number"] == 2
    assert knowledge.handoffs[0]["language"] == "sw"
    assert knowledge.handoffs[0]["title"] == "Test Title"


def test_knowledge_handoff_receives_correct_data_structure() -> None:
    """Test that knowledge handoff receives all required data fields."""
    knowledge = FakeKnowledgeHandoff()
    versions = FakeVersionRepository()
    versions.save_version(
        make_version(
            resource_id="r1",
            version_number=3,
            text="final approved text",
        )
    )
    stage = build_publish_stage(versions=versions, knowledge=knowledge)[0]

    resource = approved_resource(
        version_number=3,
        source_url="https://health.gov/swahili-guide",
        source_metadata={
            "approved_by": "reviewer-8",
            "title": "Maternal Health Guide",
            "license_id": "CC-BY-4.0",
        },
        detected_language="sw",
    )

    stage.handle(resource)

    handoff = knowledge.handoffs[0]
    assert handoff["resource_id"] == "r1"
    assert handoff["source_url"] == "https://health.gov/swahili-guide"
    assert handoff["title"] == "Maternal Health Guide"
    assert handoff["translated_text"] == "final approved text"
    assert handoff["version_number"] == 3
    assert handoff["language"] == "sw"
    assert handoff["metadata"]["license_id"] == "CC-BY-4.0"
    assert handoff["metadata"]["approved_by"] == "reviewer-8"


def test_knowledge_handoff_not_called_when_not_configured() -> None:
    """Test that knowledge handoff is skipped when not configured."""
    versions = FakeVersionRepository()
    versions.save_version(make_version(version_number=1, text="text"))
    stage, search = build_publish_stage(versions=versions, knowledge=None)

    result = stage.handle(approved_resource(version_number=1))

    assert result.next_status == ResourceStatus.PUBLISHED
    assert result.details["knowledge_handoff"] is False
    # Search index should still work
    assert "r1" in search.indexed


def test_knowledge_handoff_with_version_management() -> None:
    """Test that knowledge handoff supports version management (machine vs human)."""
    knowledge = FakeKnowledgeHandoff()
    versions = FakeVersionRepository()
    # Version 1 (machine) and Version 2 (human)
    versions.save_version(make_version(version_number=1, text="machine translation"))
    versions.save_version(make_version(version_number=2, text="human edited"))
    stage, _search = build_publish_stage(versions=versions, knowledge=knowledge)

    result = stage.handle(approved_resource(version_number=2))

    # Should handoff version 2 (human edited), not version 1
    assert knowledge.handoffs[0]["version_number"] == 2
    assert knowledge.handoffs[0]["translated_text"] == "human edited"
    assert result.details["approved_version"] == 2


def test_knowledge_handoff_preserves_search_index_functionality() -> None:
    """Test that knowledge handoff doesn't break existing search index functionality."""
    knowledge = FakeKnowledgeHandoff()
    versions = FakeVersionRepository()
    versions.save_version(make_version(version_number=1, text="searchable text"))
    stage, search = build_publish_stage(versions=versions, knowledge=knowledge)

    result = stage.handle(
        approved_resource(version_number=1, source_metadata={"title": "Searchable Title"})
    )

    # Both search index and knowledge handoff should work
    assert "r1" in search.indexed
    assert search.indexed["r1"].translated_text == "searchable text"
    assert len(knowledge.handoffs) == 1
    assert result.next_status == ResourceStatus.PUBLISHED
