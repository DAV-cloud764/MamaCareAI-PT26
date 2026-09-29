"""
Tests for sql_repositories.py using in-memory SQLite.
"""

from datetime import UTC, datetime
from typing import cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.pipeline.adapters.storage.sql_repositories import (
    AssignmentNotFoundError,
    AuditEvent,
    AuthorKind,
    Base,
    ContentVersion,
    DuplicateResourceError,
    InvalidStateTransitionError,
    NormalizedDocument,
    Resource,
    ResourceNotFoundError,
    ResourceStatus,
    ReviewAssignment,
    ReviewDecision,
    SqlDocumentRepository,
    SqlResourceRepository,
    SqlReviewRepository,
    SqlVersionRepository,
    TextBlock,
    TranslationUnit,
    UnauthorizedAssignmentAccessError,
)


@pytest.fixture
def session_factory():
    """Provides an in-memory SQLite session factory with pre-created tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    return factory


# ---------------------------------------------------------------------------
# SqlResourceRepository Tests
# ---------------------------------------------------------------------------


def test_resource_repo_add_and_get(session_factory):
    repo = SqlResourceRepository(session_factory=session_factory)
    now = datetime.now(UTC)

    resource = Resource(
        resource_id="res-1",
        source_type="pdf",
        source_url="https://example.com/doc.pdf",
        status=ResourceStatus.PENDING,
        content_hash="hash-123",
        submitted_at=now,
        updated_at=now,
        attempt_count=0,
    )

    repo.add(resource)
    retrieved = repo.get("res-1")

    assert retrieved.resource_id == "res-1"
    assert retrieved.status == ResourceStatus.PENDING
    assert retrieved.content_hash == "hash-123"
    assert retrieved.version == 1


def test_resource_repo_duplicate_content_hash_raises_error(session_factory):
    repo = SqlResourceRepository(session_factory=session_factory)
    now = datetime.now(UTC)

    res1 = Resource(
        resource_id="res-1",
        source_type="pdf",
        source_url="https://example.com/1.pdf",
        status=ResourceStatus.PENDING,
        content_hash="dup-hash",
        submitted_at=now,
        updated_at=now,
    )
    res2 = Resource(
        resource_id="res-2",
        source_type="pdf",
        source_url="https://example.com/2.pdf",
        status=ResourceStatus.PENDING,
        content_hash="dup-hash",
        submitted_at=now,
        updated_at=now,
    )

    repo.add(res1)
    with pytest.raises(DuplicateResourceError):
        repo.add(res2)


def test_resource_repo_save_updates_version_and_data(session_factory):
    repo = SqlResourceRepository(session_factory=session_factory)
    now = datetime.now(UTC)

    resource = Resource(
        resource_id="res-1",
        source_type="pdf",
        source_url="https://example.com/doc.pdf",
        status=ResourceStatus.PENDING,
        content_hash="hash-1",
        submitted_at=now,
        updated_at=now,
        version=1,
    )
    repo.add(resource)

    # Modify and save
    resource.status = ResourceStatus.PROCESSING
    repo.save(resource)

    updated = repo.get("res-1")
    assert updated.status == ResourceStatus.PROCESSING
    assert updated.version == 2


def test_resource_repo_save_stale_version_raises_concurrency_error(session_factory):
    repo = SqlResourceRepository(session_factory=session_factory)
    now = datetime.now(UTC)

    resource = Resource(
        resource_id="res-1",
        source_type="pdf",
        source_url="https://example.com/doc.pdf",
        status=ResourceStatus.PENDING,
        content_hash="hash-1",
        submitted_at=now,
        updated_at=now,
        version=1,
    )
    repo.add(resource)

    # Simulate stale state with wrong version number
    resource.version = 99
    with pytest.raises(InvalidStateTransitionError):
        repo.save(resource)


def test_resource_repo_list_by_status(session_factory):
    repo = SqlResourceRepository(session_factory=session_factory)
    now = datetime.now(UTC)

    for i in range(3):
        repo.add(
            Resource(
                resource_id=f"res-{i}",
                source_type="pdf",
                source_url=f"https://example.com/{i}.pdf",
                status=ResourceStatus.PENDING if i < 2 else ResourceStatus.COMPLETED,
                content_hash=f"hash-{i}",
                submitted_at=now,
                updated_at=now,
            )
        )

    pending_list = repo.list_by_status(ResourceStatus.PENDING)
    assert len(pending_list) == 2


# ---------------------------------------------------------------------------
# SqlDocumentRepository Tests
# ---------------------------------------------------------------------------


def test_document_repo_save_and_get(session_factory):
    res_repo = SqlResourceRepository(session_factory=session_factory)
    doc_repo = SqlDocumentRepository(session_factory=session_factory)
    now = datetime.now(UTC)

    # Parent resource must exist due to foreign key constraint
    res_repo.add(
        Resource(
            resource_id="res-doc-1",
            source_type="pdf",
            source_url="https://example.com/doc.pdf",
            status=ResourceStatus.PENDING,
            content_hash="hash-doc-1",
            submitted_at=now,
            updated_at=now,
        )
    )

    doc = NormalizedDocument(
        resource_id="res-doc-1",
        title="Sample Document",
        author="John Doe",
        published_date=cast(str, now),
        blocks=[
            TextBlock(block_id="b1", text="Header text", order=0),
            TextBlock(block_id="b2", text="Body text", order=1),
        ],
    )

    doc_repo.save_document(doc)
    retrieved = doc_repo.get_document("res-doc-1")

    assert retrieved.title == "Sample Document"
    assert len(retrieved.blocks) == 2
    assert retrieved.blocks[0].text == "Header text"
    assert retrieved.blocks[0].block_id == "b1"


# ---------------------------------------------------------------------------
# SqlVersionRepository Tests
# ---------------------------------------------------------------------------


def test_version_repo_auto_increments_version_number(session_factory):
    res_repo = SqlResourceRepository(session_factory=session_factory)
    ver_repo = SqlVersionRepository(session_factory=session_factory)
    now = datetime.now(UTC)

    res_repo.add(
        Resource(
            resource_id="res-ver-1",
            source_type="pdf",
            source_url="https://example.com/doc.pdf",
            status=ResourceStatus.PENDING,
            content_hash="hash-ver-1",
            submitted_at=now,
            updated_at=now,
        )
    )

    v1 = ContentVersion(
        version_id="v-1",
        resource_id="res-ver-1",
        version_number=0,  # Should be computed dynamically
        author_kind=AuthorKind.MACHINE,
        author_id="machine-1",
        created_at=now,
        units=[TranslationUnit(unit_id="u1", source_text="Hello", target_text="Jambo")],
    )

    v2 = ContentVersion(
        version_id="v-2",
        resource_id="res-ver-1",
        version_number=0,
        author_kind=AuthorKind.HUMAN,
        author_id="human-1",
        created_at=now,
        units=[
            TranslationUnit(unit_id="u1", source_text="Hello", target_text="Habari")
        ],
    )

    ver_repo.save_version(v1)
    ver_repo.save_version(v2)

    latest = ver_repo.get_latest("res-ver-1")
    assert latest is not None
    assert latest.version_number == 2
    assert latest.author_kind == AuthorKind.HUMAN
    assert latest.units[0].target_text == "Habari"


# ---------------------------------------------------------------------------
# SqlReviewRepository Tests
# ---------------------------------------------------------------------------


def test_review_repo_claim_next_and_save(session_factory):
    res_repo = SqlResourceRepository(session_factory=session_factory)
    rev_repo = SqlReviewRepository(session_factory=session_factory)
    now = datetime.now(UTC)

    res_repo.add(
        Resource(
            resource_id="res-rev-1",
            source_type="pdf",
            source_url="https://example.com/doc.pdf",
            status=ResourceStatus.PENDING,
            content_hash="hash-rev-1",
            submitted_at=now,
            updated_at=now,
        )
    )

    assignment = ReviewAssignment(
        assignment_id="assign-1",
        resource_id="res-rev-1",
        reviewer_id=None,
        priority=10,
        assigned_at=now,
    )
    rev_repo.create_assignment(assignment)

    # Claim next assignment
    claimed = rev_repo.claim_next(reviewer_id="reviewer-7")
    assert claimed is not None
    assert claimed.assignment_id == "assign-1"
    assert claimed.reviewer_id == "reviewer-7"

    # Complete assignment
    claimed.decision = ReviewDecision.APPROVED
    claimed.completed_at = cast(str, datetime.now(UTC))
    rev_repo.save_assignment(claimed)

    updated = rev_repo.get_assignment("assign-1")
    assert updated.decision == ReviewDecision.APPROVED


def test_review_repo_unauthorized_save_raises_error(session_factory):
    res_repo = SqlResourceRepository(session_factory=session_factory)
    rev_repo = SqlReviewRepository(session_factory=session_factory)
    now = datetime.now(UTC)

    res_repo.add(
        Resource(
            resource_id="res-rev-2",
            source_type="pdf",
            source_url="https://example.com/doc.pdf",
            status=ResourceStatus.PENDING,
            content_hash="hash-rev-2",
            submitted_at=now,
            updated_at=now,
        )
    )

    rev_repo.create_assignment(
        ReviewAssignment(
            assignment_id="assign-2",
            resource_id="res-rev-2",
            reviewer_id="user-A",
            assigned_at=now,
        )
    )

    # User B attempts to save modifications on User A's assignment
    unauthorized = ReviewAssignment(
        assignment_id="assign-2",
        resource_id="res-rev-2",
        reviewer_id="user-B",
        decision=ReviewDecision.APPROVED,
        assigned_at=now,
    )

    with pytest.raises(UnauthorizedAssignmentAccessError):
        rev_repo.save_assignment(unauthorized)


def test_review_repo_audit_trail(session_factory):
    res_repo = SqlResourceRepository(session_factory=session_factory)
    rev_repo = SqlReviewRepository(session_factory=session_factory)
    now = datetime.now(UTC)

    res_repo.add(
        Resource(
            resource_id="res-audit-1",
            source_type="pdf",
            source_url="https://example.com/doc.pdf",
            status=ResourceStatus.PENDING,
            content_hash="hash-audit-1",
            submitted_at=now,
            updated_at=now,
        )
    )

    event = AuditEvent(
        event_id="evt-1",
        resource_id="res-audit-1",
        actor_id="user-1",
        action="STATUS_CHANGE",
        from_status=ResourceStatus.PENDING,
        to_status=ResourceStatus.PROCESSING,
        at=now,
        details=cast(str, {"reason": "Started processing"}),
    )

    rev_repo.append_audit(event)
    history = rev_repo.list_audit("res-audit-1")

    assert len(history) == 1
    assert history[0].action == "STATUS_CHANGE"
    assert history[0].from_status == ResourceStatus.PENDING
    assert history[0].to_status == ResourceStatus.PROCESSING


def test_resource_repo_get_missing_raises_not_found(session_factory):
    repo = SqlResourceRepository(session_factory=session_factory)
    with pytest.raises(ResourceNotFoundError):
        repo.get("does-not-exist")


def test_review_repo_get_missing_assignment_raises_not_found(session_factory):
    repo = SqlReviewRepository(session_factory=session_factory)
    with pytest.raises(AssignmentNotFoundError):
        repo.get_assignment("does-not-exist")
