"""
Ports: repositories — structured pipeline state (PDF 3.5, relational database).

Four small interfaces instead of one big `Database` class. That is Interface
Segregation: the translate stage needs resources and versions, not review
assignments, so it depends only on what it uses — and a test fake for it only
has to implement what it uses.

Ports return DOMAIN OBJECTS, never ORM rows, dicts, or database cursors. If
`Resource` leaks a SQLAlchemy model into a stage, the stage is now coupled to
the database and the layering is broken.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..domain.enums import ResourceStatus
from ..domain.models import (
    AuditEvent,
    ContentVersion,
    NormalizedDocument,
    Resource,
    ReviewAssignment,
)


class ResourceRepository(ABC):
    """Persistence for the `Resource` control record."""

    @abstractmethod
    def add(self, resource: Resource) -> None:
        """Insert a new resource. Raise if the resource_id already exists."""
        raise NotImplementedError

    @abstractmethod
    def get(self, resource_id: str) -> Resource:
        """Load one resource. Raise a PermanentError if it does not exist."""
        raise NotImplementedError

    @abstractmethod
    def find_by_content_hash(self, content_hash: str) -> Resource | None:
        """Dedup lookup (PDF 3.1). Return None when the hash is new.

        TODO: this column needs a UNIQUE index. Without it, two workers
        submitting the same URL at the same time both see None and both
        proceed — the index is what makes dedup actually hold under concurrency.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, resource: Resource) -> None:
        """Persist an updated resource.

        TODO (IMPORTANT — concurrency): implement this as an optimistic
        conditional update, i.e.
            UPDATE resources SET status=:new WHERE id=:id AND status=:expected
        and raise `InvalidStateTransition` when zero rows are affected. A blind
        UPDATE lets two workers overwrite each other's status and a resource
        silently skips a stage. This is the single most important line of
        correctness in the persistence layer.
        """
        raise NotImplementedError

    @abstractmethod
    def list_by_status(
        self, status: ResourceStatus, *, limit: int = 100, offset: int = 0
    ) -> list[Resource]:
        """Page through resources in a given state.

        Powers the review queue and the "how much is stuck in translation?"
        operational view. Always paginated — never load a whole table.
        """
        raise NotImplementedError


class DocumentRepository(ABC):
    """Persistence for extracted, normalized documents (the PDF 3.2 schema)."""

    @abstractmethod
    def save_document(self, document: NormalizedDocument) -> None:
        """Store the normalized document for a resource. Overwrites on re-extraction."""
        raise NotImplementedError

    @abstractmethod
    def get_document(self, resource_id: str) -> NormalizedDocument:
        """Load the normalized source document. Raise if extraction has not run."""
        raise NotImplementedError


class VersionRepository(ABC):
    """Append-only history of translated content (PDF 3.5).

    APPEND-ONLY IS A HARD RULE. There is no `update_version` method on this
    interface and there must never be one. A human edit inserts version N+1;
    the machine translation at version 1 stays readable forever. That is what
    makes the MT-vs-human diff — the feedback loop's training signal and the
    audit trail — possible at all.
    """

    @abstractmethod
    def save_version(self, version: ContentVersion) -> None:
        """Insert a new version.

        TODO: assign `version_number` inside the transaction (max + 1 for that
        resource, with the row locked). Computing it in Python before the call
        gives two concurrent reviewers the same number.
        """
        raise NotImplementedError

    @abstractmethod
    def save_version_if_current(
        self, version: ContentVersion, *, base_version_number: int
    ) -> ContentVersion:
        """Insert `version` only if the resource is still at `base_version_number`.

        This is the stale-write guard for the human review loop, and it is the
        only thing standing between two reviewers' browsers and a silently
        discarded edit. The reviewer loaded version N, spent ten minutes
        correcting it, and submitted. Meanwhile a second reviewer, or the same
        one in another tab, saved version N+1. Appending N+1 now would produce
        a history where the second submission silently overwrites the first —
        and because versions are append-only, nothing would ever reveal it.
        Nothing downstream can detect the loss: both versions exist, both are
        readable, and the wrong one is the latest.

        So the check and the insert must be ONE statement against the current
        state, inside one transaction. Reading the current version, comparing
        it in Python, and then inserting is the same bug with more steps.

        Raise `StaleVersionError` (a `PermanentError`, not retryable) when the
        resource has moved on. The API layer maps that to 409 CONFLICT so the
        reviewer is told to reload rather than shown a generic failure.
        """
        raise NotImplementedError

    @abstractmethod
    def get_latest(self, resource_id: str) -> ContentVersion | None:
        """Highest version number for a resource — what gets published."""
        raise NotImplementedError

    @abstractmethod
    def get_machine_version(self, resource_id: str) -> ContentVersion | None:
        """The original MT output (version 1), for side-by-side diffing."""
        raise NotImplementedError

    @abstractmethod
    def list_versions(self, resource_id: str) -> list[ContentVersion]:
        """Full history, oldest first. Powers the version-history UI."""
        raise NotImplementedError


class ReviewRepository(ABC):
    """Review assignments and the governance audit trail (PDF 3.6)."""

    @abstractmethod
    def create_assignment(self, assignment: ReviewAssignment) -> None:
        """Put a resource into the review queue."""
        raise NotImplementedError

    @abstractmethod
    def get_assignment(self, assignment_id: str) -> ReviewAssignment:
        """Load one assignment. Raise if it does not exist."""
        raise NotImplementedError

    @abstractmethod
    def claim_next(self, reviewer_id: str) -> ReviewAssignment | None:
        """Atomically assign the highest-priority unclaimed item to a reviewer.

        TODO: must be atomic (SELECT ... FOR UPDATE SKIP LOCKED, or the SQLite
        equivalent). Two reviewers editing the same document because both
        claimed it is the classic failure of a naive review queue, and it wastes
        the scarcest resource in this whole system: reviewer time.
        """
        raise NotImplementedError

    @abstractmethod
    def save_assignment(self, assignment: ReviewAssignment) -> None:
        """Persist a decision/completion on an assignment."""
        raise NotImplementedError

    @abstractmethod
    def append_audit(self, event: AuditEvent) -> None:
        """Append one immutable audit row.

        Never updated, never deleted. If this table can be edited, we cannot
        claim an audit trail exists.
        """
        raise NotImplementedError

    @abstractmethod
    def list_audit(self, resource_id: str) -> list[AuditEvent]:
        """Full history of who did what to a resource, oldest first."""
        raise NotImplementedError
