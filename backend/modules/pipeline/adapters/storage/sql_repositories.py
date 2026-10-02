"""
SQLAlchemy repositories — the relational side of PDF 3.5.

ONE FILE, BOTH ENVIRONMENTS. SQLAlchemy speaks SQLite (free MVP) and PostgreSQL
(production), so the MVP exercises the real production code path rather than a
throwaway dev implementation. Only `PIPELINE_DATABASE_URL` changes.

THIS MODULE DEFINES NO DOMAIN TYPES.

It used to. It declared its own `ResourceStatus`, `ReviewDecision`,
`Resource`, `TextBlock`, `NormalizedDocument`, `TranslationUnit`,
`ContentVersion`, `ReviewAssignment` and `AuditEvent`, and its own copies of the
four repository interfaces. That was not duplication for its own sake — the two
vocabularies drifted until they no longer described the same pipeline:

  - the local `ResourceStatus` had five values (pending, processing, completed,
    error, failed) against the domain's seventeen. `get()` therefore raised
    `ValueError` on every status the state machine can actually produce, so no
    resource could ever be read back after leaving SUBMITTED.
  - the local `TranslationUnit` spelled the field `target_text` where the domain
    spells it `translated_text`, so `PublishStage` could not consume what this
    adapter returned.
  - the local `NormalizedDocument.published_date` was a `str` where the domain
    requires a `datetime`.
  - the repository classes inherited empty marker classes rather than the real
    abstract ports, so nothing checked that these implementations matched the
    contract the stages are written against.

Every type below is now imported from `domain/` and `ports/`. Mapping between
ORM rows and domain objects is this adapter's job and nothing more; the domain
models stay free of persistence concerns, as `domain/models.py` requires.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    exc,
    func,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, declarative_base

from ...domain.enums import (
    ResourceStatus,
    ReviewDecision,
    SourceType,
    VersionAuthorKind,
)
from ...domain.errors import (
    InvalidStateTransition,
    PermanentError,
    ResourceNotFound,
    StaleVersionError,
)
from ...domain.models import (
    AuditEvent,
    ContentVersion,
    NormalizedDocument,
    Resource,
    ReviewAssignment,
    TextBlock,
    TranslationUnit,
)
from ...ports.repositories import (
    DocumentRepository,
    ResourceRepository,
    ReviewRepository,
    VersionRepository,
)

logger = logging.getLogger(__name__)

Base = declarative_base()

# JSON type that uses JSONB on PostgreSQL and fallback JSON on SQLite
JsonType = JSON().with_variant(JSONB(), "postgresql")


# ---------------------------------------------------------------------------
# ORM TABLE DEFINITIONS
# ---------------------------------------------------------------------------
#
# EVERY timestamp column is timezone-aware. The domain stamps tz-aware UTC
# (`domain.models.utc_now`) and refuses to produce naive datetimes, but
# PostgreSQL's default TIMESTAMP WITHOUT TIME ZONE silently DROPS the offset on
# write: the value goes in as UTC and comes back naive, so a round trip quietly
# changes the type. Nothing fails — the timestamps just stop being comparable
# across workers, which is exactly what the audit trail's ordering depends on.
# SQLite ignores the flag, so this costs nothing on the dev stack.

# `content_hash` is nullable AND unique. A resource row is inserted at
# submission, before anything has been fetched, so there is a window where the
# hash genuinely does not exist yet. `nullable=False` made submission
# impossible. Unique still holds where it matters: every backend treats NULLs as
# distinct from each other, so two unhashed resources coexist but a second
# resource carrying a hash already present is rejected — which is the
# deduplication guarantee `ResourceRepository.find_by_content_hash` depends on.
#
# There is deliberately no `version` column. Optimistic concurrency is done on
# `status`, which is a complete compare-and-swap token: ALLOWED_TRANSITIONS has
# no self-loops, so every legal transition changes it.


class ResourceORM(Base):
    __tablename__ = "resources"

    resource_id = Column(String, primary_key=True)
    source_type = Column(String, nullable=False)
    source_url = Column(String, nullable=False)
    status = Column(String, nullable=False)
    content_hash = Column(String, nullable=True, unique=True)
    submitted_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    attempt_count = Column(Integer, default=0, nullable=False)
    last_error = Column(Text, nullable=True)
    raw_object_key = Column(String, nullable=True)
    detected_language = Column(String, nullable=True)
    language_confidence = Column(Float, nullable=True)
    source_metadata = Column(JsonType, nullable=True)

    __table_args__ = (Index("idx_resources_status_updated_at", "status", "updated_at"),)


class DocumentORM(Base):
    __tablename__ = "documents"

    resource_id = Column(String, ForeignKey("resources.resource_id"), primary_key=True)
    title = Column(Text, nullable=True)
    author = Column(Text, nullable=True)
    published_date = Column(DateTime(timezone=True), nullable=True)
    blocks = Column(JsonType, nullable=False)
    source_metadata = Column(JsonType, nullable=True)


class ContentVersionORM(Base):
    __tablename__ = "content_versions"

    version_id = Column(String, primary_key=True)
    resource_id = Column(
        String, ForeignKey("resources.resource_id"), nullable=False, index=True
    )
    version_number = Column(Integer, nullable=False)
    author_kind = Column(String, nullable=False)
    author_id = Column(String, nullable=True)
    engine = Column(String, nullable=True)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    units = Column(JsonType, nullable=False)

    # The append-only guarantee, enforced by the database rather than by
    # convention. A repository with an UPDATE anywhere near this table has
    # broken the design: version 1 must stay readable forever, because the
    # machine-vs-human diff is both the audit trail and the training signal.
    __table_args__ = (
        UniqueConstraint(
            "resource_id", "version_number", name="uq_resource_version_number"
        ),
    )


class ReviewAssignmentORM(Base):
    __tablename__ = "review_assignments"

    assignment_id = Column(String, primary_key=True)
    resource_id = Column(
        String, ForeignKey("resources.resource_id"), nullable=False, index=True
    )
    reviewer_id = Column(String, nullable=True)
    decision = Column(String, nullable=True)
    priority = Column(Integer, default=0, nullable=False)
    assigned_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Serves the claim query: highest priority first among unclaimed, oldest
    # first within a priority, so the most urgent document is claimed first.
    __table_args__ = (
        Index("idx_review_claim", "reviewer_id", "completed_at", text("priority DESC")),
    )


class AuditEventORM(Base):
    __tablename__ = "audit_events"

    event_id = Column(String, primary_key=True)
    resource_id = Column(
        String, ForeignKey("resources.resource_id"), nullable=False, index=True
    )
    actor_id = Column(String, nullable=False)
    action = Column(String, nullable=False)
    from_status = Column(String, nullable=True)
    to_status = Column(String, nullable=True)
    at = Column(DateTime(timezone=True), nullable=False)
    details = Column(JsonType, nullable=True)


# ---------------------------------------------------------------------------
# ROW <-> DOMAIN MAPPING
# ---------------------------------------------------------------------------
#
# The domain models are frozen dataclasses declared with `slots=True`, so they
# have no `__dict__`. Serialising them with `vars()` or `obj.__dict__` — the
# obvious one-liner — therefore fails on every one of them, and the fallback
# branch hands a live Python object to a JSON column, which the driver cannot
# encode. Every mapping below is written out explicitly instead, which also
# makes it obvious what a row actually contains.


def _to_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _from_iso(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _block_to_json(block: TextBlock) -> dict[str, Any]:
    return {
        "order": block.order,
        "kind": block.kind,
        "text": block.text,
        "start_seconds": block.start_seconds,
        "end_seconds": block.end_seconds,
    }


def _block_from_json(raw: dict[str, Any]) -> TextBlock:
    return TextBlock(
        order=raw["order"],
        kind=raw["kind"],
        text=raw["text"],
        start_seconds=raw.get("start_seconds"),
        end_seconds=raw.get("end_seconds"),
    )


def _unit_to_json(unit: TranslationUnit) -> dict[str, Any]:
    return {
        "order": unit.order,
        "source_text": unit.source_text,
        "translated_text": unit.translated_text,
        "kind": unit.kind,
        "confidence": unit.confidence,
    }


def _unit_from_json(raw: dict[str, Any]) -> TranslationUnit:
    return TranslationUnit(
        order=raw["order"],
        source_text=raw["source_text"],
        translated_text=raw["translated_text"],
        kind=raw.get("kind", "paragraph"),
        confidence=raw.get("confidence"),
    )


def _resource_to_domain(row: ResourceORM) -> Resource:
    return Resource(
        resource_id=row.resource_id,
        source_type=SourceType(row.source_type),
        source_url=row.source_url,
        status=ResourceStatus(row.status),
        content_hash=row.content_hash,
        submitted_at=_from_iso(row.submitted_at),
        updated_at=_from_iso(row.updated_at),
        attempt_count=row.attempt_count,
        last_error=row.last_error,
        raw_object_key=row.raw_object_key,
        detected_language=row.detected_language,
        language_confidence=row.language_confidence,
        source_metadata=row.source_metadata or {},
    )


def _document_to_domain(row: DocumentORM) -> NormalizedDocument:
    return NormalizedDocument(
        resource_id=row.resource_id,
        title=row.title,
        author=row.author,
        published_date=_from_iso(row.published_date),
        blocks=tuple(
            sorted(
                (_block_from_json(b) for b in (row.blocks or [])),
                key=lambda b: b.order,
            )
        ),
        source_metadata=row.source_metadata or {},
    )


def _version_to_domain(row: ContentVersionORM) -> ContentVersion:
    return ContentVersion(
        version_id=row.version_id,
        resource_id=row.resource_id,
        version_number=row.version_number,
        author_kind=VersionAuthorKind(row.author_kind),
        author_id=row.author_id,
        units=tuple(_unit_from_json(u) for u in (row.units or [])),
        created_at=_from_iso(row.created_at),
        engine=row.engine,
        note=row.note,
    )


def _assignment_to_domain(row: ReviewAssignmentORM) -> ReviewAssignment:
    return ReviewAssignment(
        assignment_id=row.assignment_id,
        resource_id=row.resource_id,
        reviewer_id=row.reviewer_id,
        decision=ReviewDecision(row.decision) if row.decision else None,
        assigned_at=_from_iso(row.assigned_at),
        completed_at=_from_iso(row.completed_at),
        priority=row.priority,
    )


def _audit_to_domain(row: AuditEventORM) -> AuditEvent:
    return AuditEvent(
        event_id=row.event_id,
        resource_id=row.resource_id,
        actor_id=row.actor_id,
        action=row.action,
        from_status=ResourceStatus(row.from_status) if row.from_status else None,
        to_status=ResourceStatus(row.to_status) if row.to_status else None,
        at=_from_iso(row.at),
        details=row.details or {},
    )


# ---------------------------------------------------------------------------
# REPOSITORY IMPLEMENTATIONS
# ---------------------------------------------------------------------------


class SqlResourceRepository(ResourceRepository):
    """Resource state in a relational database."""

    def __init__(self, *, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def add(self, resource: Resource) -> None:
        session: Session = self._session_factory()
        try:
            session.add(
                ResourceORM(
                    resource_id=resource.resource_id,
                    source_type=resource.source_type.value,
                    source_url=resource.source_url,
                    status=resource.status.value,
                    content_hash=resource.content_hash,
                    submitted_at=resource.submitted_at,
                    updated_at=resource.updated_at,
                    attempt_count=resource.attempt_count,
                    last_error=resource.last_error,
                    raw_object_key=resource.raw_object_key,
                    detected_language=resource.detected_language,
                    language_confidence=resource.language_confidence,
                    source_metadata=resource.source_metadata,
                )
            )
            session.commit()
        except exc.IntegrityError as err:
            session.rollback()
            raise PermanentError(
                f"Resource {resource.resource_id!r} already exists, or its "
                f"content_hash is already registered.",
                resource_id=resource.resource_id,
            ) from err
        finally:
            session.close()

    def get(self, resource_id: str) -> Resource:
        session: Session = self._session_factory()
        try:
            row = session.get(ResourceORM, resource_id)
            if row is None:
                raise ResourceNotFound(
                    f"Resource {resource_id!r} not found.", resource_id=resource_id
                )
            return _resource_to_domain(row)
        finally:
            session.close()

    def find_by_content_hash(self, content_hash: str) -> Resource | None:
        session: Session = self._session_factory()
        try:
            stmt = select(ResourceORM).where(ResourceORM.content_hash == content_hash)
            row = session.execute(stmt).scalar_one_or_none()
            return _resource_to_domain(row) if row is not None else None
        finally:
            session.close()

    def save(self, resource: Resource, *, expected_status: ResourceStatus) -> None:
        """Persist an updated resource, conditional on `expected_status`.

        The comparison uses the status the CALLER read, never the status found
        in the row. Reading the row first and comparing against that would
        match by construction — the predicate would always be true, and a check
        that can never fail is not a check. The one thing worth reading first
        is existence, so a missing row reports "not found" rather than the much
        more confusing "concurrent modification".
        """
        session: Session = self._session_factory()
        try:
            if session.get(ResourceORM, resource.resource_id) is None:
                raise ResourceNotFound(
                    f"Resource {resource.resource_id!r} not found.",
                    resource_id=resource.resource_id,
                )
            if expected_status is resource.status:
                logger.warning(
                    "save() for %s was handed expected_status equal to the new "
                    "status (%s), so the conditional update cannot detect a "
                    "lost update.",
                    resource.resource_id,
                    resource.status.value,
                )

            result = session.execute(
                update(ResourceORM)
                .where(
                    ResourceORM.resource_id == resource.resource_id,
                    ResourceORM.status == expected_status.value,
                )
                .values(
                    source_type=resource.source_type.value,
                    source_url=resource.source_url,
                    status=resource.status.value,
                    content_hash=resource.content_hash,
                    submitted_at=resource.submitted_at,
                    updated_at=resource.updated_at,
                    attempt_count=resource.attempt_count,
                    last_error=resource.last_error,
                    raw_object_key=resource.raw_object_key,
                    detected_language=resource.detected_language,
                    language_confidence=resource.language_confidence,
                    source_metadata=resource.source_metadata,
                )
            )
            session.commit()
            if result.rowcount == 0:
                raise InvalidStateTransition(
                    f"Concurrent modification of resource "
                    f"{resource.resource_id!r}: expected status "
                    f"{expected_status.value}, no row matched.",
                    resource_id=resource.resource_id,
                )
        except (exc.IntegrityError, InvalidStateTransition, PermanentError):
            session.rollback()
            raise
        finally:
            session.close()

    def list_by_status(
        self, status: ResourceStatus, *, limit: int = 100, offset: int = 0
    ) -> list[Resource]:
        # Never load a whole table: this powers the review queue and the
        # "what is stuck in translation" view, and both are unbounded.
        limit = min(limit, 500)
        session: Session = self._session_factory()
        try:
            stmt = (
                select(ResourceORM)
                .where(ResourceORM.status == status.value)
                .order_by(ResourceORM.updated_at.asc())
                .offset(offset)
                .limit(limit)
            )
            rows = session.execute(stmt).scalars().all()
            return [_resource_to_domain(row) for row in rows]
        finally:
            session.close()


class SqlDocumentRepository(DocumentRepository):
    """Normalized documents in a relational database."""

    def __init__(self, *, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def save_document(self, document: NormalizedDocument) -> None:
        """Store the normalized document, overwriting on re-extraction.

        Re-running the extract stage is normal (at-least-once delivery), so
        this is an upsert keyed on resource_id rather than an insert.
        """
        session: Session = self._session_factory()
        try:
            blocks = [_block_to_json(block) for block in document.blocks]
            row = session.get(DocumentORM, document.resource_id)
            if row is not None:
                row.title = document.title
                row.author = document.author
                row.published_date = document.published_date
                row.blocks = blocks
                row.source_metadata = document.source_metadata
            else:
                session.add(
                    DocumentORM(
                        resource_id=document.resource_id,
                        title=document.title,
                        author=document.author,
                        published_date=document.published_date,
                        blocks=blocks,
                        source_metadata=document.source_metadata,
                    )
                )
            session.commit()
        finally:
            session.close()

    def get_document(self, resource_id: str) -> NormalizedDocument:
        session: Session = self._session_factory()
        try:
            row = session.get(DocumentORM, resource_id)
            if row is None:
                raise ResourceNotFound(
                    f"No extracted document for resource {resource_id!r}; "
                    "extraction has not run.",
                    resource_id=resource_id,
                )
            return _document_to_domain(row)
        finally:
            session.close()


class SqlVersionRepository(VersionRepository):
    """Append-only content versions."""

    def __init__(self, *, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def save_version(self, version: ContentVersion) -> None:
        session: Session = self._session_factory()
        try:
            session.add(self._build_row(session, version))
            session.commit()
        except exc.IntegrityError as err:
            session.rollback()
            raise PermanentError(
                f"Version number collision for resource {version.resource_id!r}.",
                resource_id=version.resource_id,
            ) from err
        finally:
            session.close()

    def save_version_if_current(
        self, version: ContentVersion, *, base_version_number: int
    ) -> ContentVersion:
        """Append only if the resource is still at `base_version_number`.

        The read of the current version number and the insert share one
        transaction, and the current number is read under a row lock where the
        backend supports it. Reading the number, comparing it in Python, and
        then inserting would leave the same window open and additionally lose
        the race to a reviewer whose browser submitted a second earlier.
        """
        session: Session = self._session_factory()
        try:
            current = self._current_version_number(session, version.resource_id)
            if current != base_version_number:
                raise StaleVersionError(
                    f"Resource {version.resource_id!r} is at version {current}, "
                    f"not {base_version_number}.",
                    resource_id=version.resource_id,
                    base_version_number=base_version_number,
                    current_version_number=current,
                )
            row = self._build_row(session, version, version_number=current + 1)
            session.add(row)
            session.commit()
            return _version_to_domain(row)
        finally:
            session.close()

    def get_latest(self, resource_id: str) -> ContentVersion | None:
        session: Session = self._session_factory()
        try:
            stmt = (
                select(ContentVersionORM)
                .where(ContentVersionORM.resource_id == resource_id)
                .order_by(ContentVersionORM.version_number.desc())
                .limit(1)
            )
            row = session.execute(stmt).scalar_one_or_none()
            return _version_to_domain(row) if row is not None else None
        finally:
            session.close()

    def get_machine_version(self, resource_id: str) -> ContentVersion | None:
        session: Session = self._session_factory()
        try:
            stmt = (
                select(ContentVersionORM)
                .where(
                    ContentVersionORM.resource_id == resource_id,
                    ContentVersionORM.author_kind == VersionAuthorKind.MACHINE.value,
                )
                .order_by(ContentVersionORM.version_number.asc())
                .limit(1)
            )
            row = session.execute(stmt).scalar_one_or_none()
            return _version_to_domain(row) if row is not None else None
        finally:
            session.close()

    def list_versions(self, resource_id: str) -> list[ContentVersion]:
        session: Session = self._session_factory()
        try:
            stmt = (
                select(ContentVersionORM)
                .where(ContentVersionORM.resource_id == resource_id)
                .order_by(ContentVersionORM.version_number.asc())
            )
            rows = session.execute(stmt).scalars().all()
            return [_version_to_domain(row) for row in rows]
        finally:
            session.close()

    def _current_version_number(self, session: Session, resource_id: str) -> int:
        """Highest stored version number for a resource; 0 when there is none.

        Computed by the database rather than in Python. Doing it in Python means
        two concurrent reviewers both read N and both write N+1, and the
        UNIQUE constraint turns that into an IntegrityError for one of them
        instead of a silent duplicate.
        """
        stmt = select(
            func.coalesce(func.max(ContentVersionORM.version_number), 0)
        ).where(ContentVersionORM.resource_id == resource_id)
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            stmt = stmt.with_for_update()
        return session.execute(stmt).scalar() or 0

    def _build_row(
        self,
        session: Session,
        version: ContentVersion,
        *,
        version_number: int | None = None,
    ) -> ContentVersionORM:
        return ContentVersionORM(
            version_id=version.version_id,
            resource_id=version.resource_id,
            version_number=(
                version_number
                if version_number is not None
                else self._current_version_number(session, version.resource_id) + 1
            ),
            author_kind=version.author_kind.value,
            author_id=version.author_id,
            engine=version.engine,
            note=version.note,
            created_at=version.created_at,
            units=[_unit_to_json(unit) for unit in version.units],
        )


class SqlReviewRepository(ReviewRepository):
    """Review assignments and the append-only audit trail."""

    def __init__(self, *, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def create_assignment(self, assignment: ReviewAssignment) -> None:
        session: Session = self._session_factory()
        try:
            session.add(
                ReviewAssignmentORM(
                    assignment_id=assignment.assignment_id,
                    resource_id=assignment.resource_id,
                    reviewer_id=assignment.reviewer_id,
                    decision=assignment.decision.value
                    if assignment.decision
                    else None,
                    priority=assignment.priority,
                    assigned_at=assignment.assigned_at,
                    completed_at=assignment.completed_at,
                )
            )
            session.commit()
        except exc.IntegrityError as err:
            session.rollback()
            raise PermanentError(
                f"Review assignment {assignment.assignment_id!r} already exists.",
                resource_id=assignment.resource_id,
            ) from err
        finally:
            session.close()

    def get_assignment(self, assignment_id: str) -> ReviewAssignment:
        session: Session = self._session_factory()
        try:
            row = session.get(ReviewAssignmentORM, assignment_id)
            if row is None:
                raise ResourceNotFound(
                    f"Review assignment {assignment_id!r} not found."
                )
            return _assignment_to_domain(row)
        finally:
            session.close()

    def claim_next(self, reviewer_id: str) -> ReviewAssignment | None:
        """Atomically claim the highest-priority unclaimed assignment.

        Two reviewers reaching the review queue at the same time is the normal
        case, not an edge case, and reviewer time is the scarcest resource in
        this system. Reading the top row and then writing the claim leaves a
        window in which both reviewers read the same row and both believe they
        own it, so the claim and the read are one statement.

        PostgreSQL gets `FOR UPDATE SKIP LOCKED`, which skips rows another
        transaction already holds rather than blocking behind them. SQLite has
        no row locking, so it gets an optimistic update instead: try each
        candidate and keep the one whose `reviewer_id IS NULL` predicate still
        matched. `rowcount == 0` is the tell — someone else took it.
        """
        session: Session = self._session_factory()
        try:
            if session.bind is not None and session.bind.dialect.name == "postgresql":
                stmt = (
                    select(ReviewAssignmentORM)
                    .where(
                        ReviewAssignmentORM.reviewer_id.is_(None),
                        ReviewAssignmentORM.completed_at.is_(None),
                    )
                    .order_by(
                        ReviewAssignmentORM.priority.desc(),
                        ReviewAssignmentORM.assigned_at.asc(),
                    )
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                row = session.execute(stmt).scalar_one_or_none()
                if row is None:
                    return None
                row.reviewer_id = reviewer_id
                session.commit()
                return _assignment_to_domain(row)

            candidates = session.execute(
                select(ReviewAssignmentORM.assignment_id)
                .where(
                    ReviewAssignmentORM.reviewer_id.is_(None),
                    ReviewAssignmentORM.completed_at.is_(None),
                )
                .order_by(
                    ReviewAssignmentORM.priority.desc(),
                    ReviewAssignmentORM.assigned_at.asc(),
                )
            ).scalars().all()

            for assignment_id in candidates:
                result = session.execute(
                    update(ReviewAssignmentORM)
                    .where(
                        ReviewAssignmentORM.assignment_id == assignment_id,
                        ReviewAssignmentORM.reviewer_id.is_(None),
                    )
                    .values(reviewer_id=reviewer_id)
                )
                session.commit()
                if result.rowcount > 0:
                    row = session.get(ReviewAssignmentORM, assignment_id)
                    return _assignment_to_domain(row)
            return None
        finally:
            session.close()

    def save_assignment(self, assignment: ReviewAssignment) -> None:
        session: Session = self._session_factory()
        try:
            row = session.get(ReviewAssignmentORM, assignment.assignment_id)
            if row is None:
                raise ResourceNotFound(
                    f"Review assignment {assignment.assignment_id!r} not found."
                )
            if row.reviewer_id != assignment.reviewer_id:
                raise PermanentError(
                    f"Assignment {assignment.assignment_id!r} belongs to "
                    f"{row.reviewer_id!r}, not {assignment.reviewer_id!r}.",
                    resource_id=assignment.resource_id,
                )
            row.decision = (
                assignment.decision.value if assignment.decision else None
            )
            row.completed_at = assignment.completed_at
            session.commit()
        finally:
            session.close()

    def append_audit(self, event: AuditEvent) -> None:
        """Insert one immutable audit row. Never updated, never deleted."""
        session: Session = self._session_factory()
        try:
            session.add(
                AuditEventORM(
                    event_id=event.event_id,
                    resource_id=event.resource_id,
                    actor_id=event.actor_id,
                    action=event.action,
                    from_status=event.from_status.value
                    if event.from_status
                    else None,
                    to_status=event.to_status.value if event.to_status else None,
                    at=event.at,
                    details=event.details,
                )
            )
            session.commit()
        finally:
            session.close()

    def list_audit(self, resource_id: str) -> list[AuditEvent]:
        session: Session = self._session_factory()
        try:
            stmt = (
                select(AuditEventORM)
                .where(AuditEventORM.resource_id == resource_id)
                .order_by(AuditEventORM.at.asc())
            )
            rows = session.execute(stmt).scalars().all()
            return [_audit_to_domain(row) for row in rows]
        finally:
            session.close()
