"""
Domain models — the common schema every stage speaks.

THE KEY IDEA (PDF 3.2)
A web page, a PDF, and a video transcript all converge into ONE shape:
`NormalizedDocument`. After extraction, nothing downstream is allowed to care
where the content came from. Translation does not branch on "if this was a
PDF". Review does not branch on "if this was a video".

If you ever need `if resource.source_type == ...` inside stages/ after the
extraction stage, that is a design smell — the difference belongs in an
adapter, not in a use case.

These are frozen dataclasses on purpose: a stage produces a NEW object rather
than mutating a shared one, so a half-finished mutation can never be written
to the database when a later step throws.

TODO (junior dev):
  [ ] Keep these free of persistence concerns. No `to_sql()`, no `__tablename__`.
      Mapping to database rows is the repository adapter's job, not the model's.
  [ ] Add fields when a real feature needs them. An unused field is a lie about
      what the pipeline guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from .enums import (
    JobStatus,
    ResourceStatus,
    ReviewDecision,
    SourceType,
    VersionAuthorKind,
)


def utc_now() -> datetime:
    """Single source of time for the domain.

    Always timezone-aware UTC. Never `datetime.now()` without a tz — naive
    timestamps from different workers cannot be ordered reliably, and this
    pipeline's audit trail depends on ordering being correct.

    TODO: tests should inject a fixed clock rather than patching this globally.
    """
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Resource:
    """One submitted item, tracked from submission through publication.

    This is the row in PostgreSQL described in PDF 3.1 — the pipeline's
    control record. It holds STATE and POINTERS, never the content bytes
    themselves (those live in object storage).
    """

    resource_id: str  # UUID4 string, generated at submission
    source_type: SourceType
    source_url: str
    status: ResourceStatus
    content_hash: str | None = None  # dedup key: sha256(url + normalized content)
    submitted_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    # Pointers into object storage, not the payloads themselves.
    raw_object_key: str | None = None  # original PDF / HTML snapshot / audio

    # Language detection results (stage 3)
    detected_language: str | None = None  # ISO 639-1/639-3, e.g. "en", "fr"
    language_confidence: float | None = None

    # Bookkeeping
    attempt_count: int = 0
    last_error: str | None = None
    # Free-form provenance from the source platform (title, author, publisher...).
    source_metadata: dict[str, object] = field(default_factory=dict)

    def with_status(self, status: ResourceStatus, **changes: object) -> Resource:
        """Return a copy in a new status, stamping `updated_at`.

        Use this instead of mutating. It keeps the audit trail honest and makes
        every state change a single, greppable expression:

            resource = resource.with_status(ResourceStatus.EXTRACTED)

        NOTE: this does NOT validate the transition — call
        `state_machine.assert_can_transition()` first. Validation lives there so
        there is exactly one copy of the rules.
        """
        return replace(self, status=status, updated_at=utc_now(), **changes)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TextBlock:
    """One structural unit of a document: a heading, a paragraph, a list item.

    WHY WE DO NOT JUST USE ONE BIG STRING (PDF 3.4)
    Translation chunking must not flatten structure. A reviewer comparing
    source and translation side by side needs the headings to still line up.
    Keeping blocks lets us translate chunk-by-chunk and reassemble in order
    without losing the shape of the document.
    """

    order: int  # position in the document, 0-based
    kind: str  # "heading" | "paragraph" | "list_item" | "caption"
    text: str
    # For video transcripts only: where in the media this block appears.
    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    """The common schema from PDF 3.2. Output of extraction, input to everything after.

    `blocks` is the structured body; `raw_text` is the flattened convenience
    view used for language detection and search indexing.
    """

    resource_id: str
    title: str | None
    author: str | None
    published_date: datetime | None
    blocks: tuple[TextBlock, ...]
    source_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def raw_text(self) -> str:
        """Flattened plain text. Convenience only — never persist this as the source of truth.

        Derived from `blocks` so the two can never drift apart. This is the
        input language detection and search indexing consume, which is why a
        blank line separates blocks: a detector reading "somesymptom" out of
        two unrelated blocks joined without a gap would be reading text the
        document never contained.
        """
        return "\n\n".join(block.text for block in self.blocks)


@dataclass(frozen=True, slots=True)
class TranslationUnit:
    """One block's translation, kept aligned to its source block by `order`.

    Alignment is what makes the side-by-side review UI (PDF 3.6) possible.
    Lose `order` and you lose the ability to show a reviewer which Swahili
    paragraph corresponds to which source paragraph.
    """

    order: int
    source_text: str
    translated_text: str
    kind: str = "paragraph"
    confidence: float | None = None  # provider-supplied, when available


@dataclass(frozen=True, slots=True)
class ContentVersion:
    """An immutable version of the translated content (PDF 3.5).

    NON-NEGOTIABLE: a human edit creates a NEW version. It never overwrites the
    machine translation. Both must remain visible forever — the MT-vs-human
    diff is both the audit trail and the training signal for the feedback loop.

    A repository that implements `save_version` with an UPDATE statement has
    broken the whole design. It is INSERT-only.
    """

    version_id: str
    resource_id: str
    version_number: int  # 1 = machine translation, 2+ = human edits
    author_kind: VersionAuthorKind
    author_id: str | None  # reviewer user id; None for machine
    units: tuple[TranslationUnit, ...]
    created_at: datetime = field(default_factory=utc_now)
    engine: str | None = None  # e.g. "nllb-200-distilled-600M", "google-v3"
    note: str | None = None  # reviewer's comment on what they changed


@dataclass(frozen=True, slots=True)
class ReviewAssignment:
    """A review task: which resource, which reviewer, what they decided."""

    assignment_id: str
    resource_id: str
    reviewer_id: str | None  # None while unassigned in the queue
    decision: ReviewDecision | None = None
    assigned_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    priority: int = 0  # higher = review sooner (see review_service.py)


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """One row of the "who changed what and when" trail (PDF 3.6, Governance).

    Append-only. Written on every state transition and every review action.
    This content is treated as authoritative once published, so "we could not
    reconstruct who approved it" is not an acceptable answer.
    """

    event_id: str
    resource_id: str
    actor_id: str  # user id, or "system:<stage_name>"
    action: str  # "transition", "edit", "approve", "publish"
    from_status: ResourceStatus | None
    to_status: ResourceStatus | None
    at: datetime = field(default_factory=utc_now)
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Job:
    """One unit of work on the queue.

    Deliberately tiny: an ID and a stage name, never the document payload.
    Queues have message size limits and payloads go stale; the worker always
    re-reads current state from the repository. This also makes retries safe —
    a retried job re-reads fresh state instead of replaying a stale snapshot.
    """

    job_id: str
    resource_id: str
    stage: str  # target stage name, e.g. "extract"
    status: JobStatus = JobStatus.PENDING
    attempts: int = 0
    enqueued_at: datetime = field(default_factory=utc_now)
    # Set when a retry must not run before a given time (provider backoff).
    not_before: datetime | None = None
