"""
SqlAlchemy repository tests, written against the canonical domain contract.

These tests used to import every domain type from the adapter they were
testing, which is how a repository could return objects no stage can consume
and still show twelve green tests. The vocabulary is now `domain.enums` and
`domain.models`, so a drift between the two shows up here as a failure rather
than as a surprise in a stage.

Statuses used are real pipeline statuses taken from the happy path in
`state_machine.ALLOWED_TRANSITIONS`, not invented ones, because the previous
version of this file used a five-value `ResourceStatus` that the state machine
cannot produce.

Backends: in-memory SQLite for the single-threaded contract tests, and a
file-backed SQLite for the concurrency test — an in-memory database gives each
thread its own private copy, so two threads would not be racing on one row at
all, and the test would pass without proving anything.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.pipeline.adapters.storage.sql_repositories import (
    Base,
    SqlDocumentRepository,
    SqlResourceRepository,
    SqlReviewRepository,
    SqlVersionRepository,
)
from modules.pipeline.domain.enums import (
    ResourceStatus,
    ReviewDecision,
    SourceType,
    VersionAuthorKind,
)
from modules.pipeline.domain.errors import (
    InvalidStateTransition,
    PermanentError,
    StaleVersionError,
)
from modules.pipeline.domain.models import (
    AuditEvent,
    ContentVersion,
    NormalizedDocument,
    Resource,
    ReviewAssignment,
    TextBlock,
    TranslationUnit,
)

NOW = datetime(2026, 5, 10, 9, 30, tzinfo=UTC)


@pytest.fixture
def session_factory():
    """In-memory SQLite with the tables created. Single-threaded tests only."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def threaded_session_factory(tmp_path):
    """File-backed SQLite shared across threads.

    A second thread gets its own connection, so the engine has to be told that
    sessions may be created off the main thread.
    """
    engine = create_engine(
        f"sqlite:///{tmp_path / 'concurrency.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def make_resource(
    resource_id: str = "res-1",
    *,
    status: ResourceStatus = ResourceStatus.SUBMITTED,
    content_hash: str | None = None,
) -> Resource:
    """A submitted resource. No content hash unless a test asks for one.

    Defaulting the hash to a constant made every helper-built resource collide
    on the unique index, which is a legitimate failure but not the one these
    tests are about.
    """
    return Resource(
        resource_id=resource_id,
        source_type=SourceType.WEB,
        source_url=f"https://example.tz/{resource_id}",
        status=status,
        content_hash=content_hash,
        submitted_at=NOW,
        updated_at=NOW,
    )


# ---------------------------------------------------------------------------
# SqlResourceRepository
# ---------------------------------------------------------------------------


def test_resource_round_trips_through_the_database(session_factory):
    repo = SqlResourceRepository(session_factory=session_factory)
    repo.add(make_resource(content_hash="hash-1"))

    stored = repo.get("res-1")

    assert stored.resource_id == "res-1"
    assert stored.source_type is SourceType.WEB
    assert stored.status is ResourceStatus.SUBMITTED
    assert stored.content_hash == "hash-1"


def test_every_pipeline_status_survives_a_round_trip(session_factory):
    """The reason this file was rewritten.

    The adapter used to carry its own five-value `ResourceStatus`, so reading
    back any status the state machine can actually produce raised ValueError.
    Every status in the enum is checked here so that class of drift cannot come
    back unnoticed.
    """
    repo = SqlResourceRepository(session_factory=session_factory)
    for index, status in enumerate(ResourceStatus):
        repo.add(make_resource(f"res-{index}", status=status))

    for index, status in enumerate(ResourceStatus):
        assert repo.get(f"res-{index}").status is status


def test_a_resource_may_be_added_before_anything_is_fetched(session_factory):
    """`content_hash` is nullable on purpose.

    A row is inserted at submission, before the fetcher has run, so there is a
    real window where no hash exists. Making the column NOT NULL made
    submission impossible.
    """
    repo = SqlResourceRepository(session_factory=session_factory)
    repo.add(make_resource(content_hash=None))

    assert repo.get("res-1").content_hash is None


def test_several_unhashed_resources_can_coexist(session_factory):
    """Unique-when-present, not unique-always.

    Every backend treats NULLs as distinct from each other, so the index stops
    deduplication without blocking submission.
    """
    repo = SqlResourceRepository(session_factory=session_factory)
    repo.add(make_resource("res-1", content_hash=None))
    repo.add(make_resource("res-2", content_hash=None))

    assert repo.get("res-1").content_hash is None
    assert repo.get("res-2").content_hash is None


def test_duplicate_content_hash_is_rejected(session_factory):
    repo = SqlResourceRepository(session_factory=session_factory)
    repo.add(make_resource("res-1", content_hash="shared"))

    with pytest.raises(PermanentError):
        repo.add(make_resource("res-2", content_hash="shared"))


def test_find_by_content_hash_locates_the_resource(session_factory):
    repo = SqlResourceRepository(session_factory=session_factory)
    repo.add(make_resource(content_hash="findable"))

    found = repo.find_by_content_hash("findable")

    assert found is not None
    assert found.resource_id == "res-1"
    assert repo.find_by_content_hash("absent") is None


def test_save_persists_a_status_transition(session_factory):
    repo = SqlResourceRepository(session_factory=session_factory)
    repo.add(make_resource())

    repo.save(
        make_resource().with_status(ResourceStatus.FETCHED),
        expected_status=ResourceStatus.SUBMITTED,
    )

    assert repo.get("res-1").status is ResourceStatus.FETCHED


def test_get_missing_resource_raises_permanent_error(session_factory):
    repo = SqlResourceRepository(session_factory=session_factory)

    with pytest.raises(PermanentError, match="not found"):
        repo.get("does-not-exist")


def test_list_by_status_filters_and_paginates(session_factory):
    repo = SqlResourceRepository(session_factory=session_factory)
    repo.add(make_resource("res-1", status=ResourceStatus.SUBMITTED))
    repo.add(make_resource("res-2", status=ResourceStatus.SUBMITTED))
    repo.add(make_resource("res-3", status=ResourceStatus.PUBLISHED))

    submitted = repo.list_by_status(ResourceStatus.SUBMITTED)

    assert len(submitted) == 2
    assert len(repo.list_by_status(ResourceStatus.SUBMITTED, limit=1)) == 1
    assert repo.list_by_status(ResourceStatus.SUBMITTED, offset=1) != []
    assert repo.list_by_status(ResourceStatus.DUPLICATE) == []


# ---------------------------------------------------------------------------
# Concurrency — PIPELINE_BACKLOG.md:200
# ---------------------------------------------------------------------------
#
# "`save()` must be a conditional update and `claim_next()` must be atomic —
# both pass every single-threaded test while being broken. It needs a
# concurrency test that spawns two threads and asserts exactly one wins."
#
# This is that test. The barrier makes the race deterministic rather than
# lucky: both readers observe the same status before either writes, which is
# the interleaving a single-threaded test can never produce.


def test_two_writers_racing_the_same_status_produce_exactly_one_winner(
    threaded_session_factory,
):
    """PIPELINE_BACKLOG.md:200 — "asserts exactly one wins".

    The barrier makes the interleaving deterministic rather than lucky: both
    threads read the resource as SUBMITTED before either writes, which is the
    window a single-threaded test can never open.

    Honest scope: SQLite serialises writes, so part of what this exercises is
    the database's own locking. What it proves is that the conditional update
    refuses the loser instead of overwriting the winner — and a blind UPDATE
    would let both "succeed" here, which is the bug.
    """
    SqlResourceRepository(session_factory=threaded_session_factory).add(make_resource())

    both_have_read = Barrier(2)

    def read_then_write() -> str:
        repo = SqlResourceRepository(session_factory=threaded_session_factory)
        stale = repo.get("res-1")
        both_have_read.wait(timeout=10)
        try:
            repo.save(
                stale.with_status(ResourceStatus.FETCHED),
                expected_status=ResourceStatus.SUBMITTED,
            )
        except InvalidStateTransition:
            return "lost"
        return "won"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [
            future.result() for future in [pool.submit(read_then_write) for _ in range(2)]
        ]

    assert sorted(outcomes) == ["lost", "won"]
    assert (
        SqlResourceRepository(session_factory=threaded_session_factory)
        .get("res-1")
        .status
        is ResourceStatus.FETCHED
    )


def test_a_lost_update_is_refused_rather_than_silently_applied(session_factory):
    """The single-writer version of the same rule, so a regression is obvious.

    Silently applying the second write is the failure this guards: the row
    exists, its status is legal, and its content is simply wrong, with nothing
    anywhere reporting a problem.
    """
    repo = SqlResourceRepository(session_factory=session_factory)
    repo.add(make_resource())

    stale = repo.get("res-1")
    repo.save(
        stale.with_status(ResourceStatus.FETCHED),
        expected_status=ResourceStatus.SUBMITTED,
    )

    with pytest.raises(InvalidStateTransition):
        repo.save(
            stale.with_status(ResourceStatus.DUPLICATE),
            expected_status=ResourceStatus.SUBMITTED,
        )

    assert repo.get("res-1").status is ResourceStatus.FETCHED


# ---------------------------------------------------------------------------
# SqlDocumentRepository
# ---------------------------------------------------------------------------


def test_document_blocks_round_trip_in_order(session_factory):
    SqlResourceRepository(session_factory=session_factory).add(make_resource())
    documents = SqlDocumentRepository(session_factory=session_factory)

    documents.save_document(
        NormalizedDocument(
            resource_id="res-1",
            title="Dalili za hatari",
            author="Wizara ya Afya",
            published_date=NOW,
            blocks=(
                TextBlock(order=0, kind="heading", text="Dalili za hatari"),
                TextBlock(order=1, kind="paragraph", text="Nenda kituo cha afya."),
            ),
        )
    )

    stored = documents.get_document("res-1")

    assert stored.title == "Dalili za hatari"
    assert [b.order for b in stored.blocks] == [0, 1]
    assert stored.blocks[0].kind == "heading"
    assert stored.blocks[0].text == "Dalili za hatari"
    assert stored.published_date is not None


def test_re_extraction_overwrites_the_document(session_factory):
    """At-least-once delivery means extract runs twice. That must be safe."""
    SqlResourceRepository(session_factory=session_factory).add(make_resource())
    documents = SqlDocumentRepository(session_factory=session_factory)

    for title in ("Kwanza", "Pili"):
        documents.save_document(
            NormalizedDocument(
                resource_id="res-1",
                title=title,
                author=None,
                published_date=None,
                blocks=(TextBlock(order=0, kind="paragraph", text=title),),
            )
        )

    assert documents.get_document("res-1").title == "Pili"


def test_get_missing_document_raises_permanent_error(session_factory):
    documents = SqlDocumentRepository(session_factory=session_factory)

    with pytest.raises(PermanentError, match="extraction has not run"):
        documents.get_document("res-1")


# ---------------------------------------------------------------------------
# SqlVersionRepository
# ---------------------------------------------------------------------------


def make_version(
    version_id: str,
    resource_id: str = "res-1",
    *,
    author_kind: VersionAuthorKind = VersionAuthorKind.MACHINE,
    author_id: str | None = None,
    translated_text: str = "Jambo",
) -> ContentVersion:
    return ContentVersion(
        version_id=version_id,
        resource_id=resource_id,
        version_number=0,
        author_kind=author_kind,
        author_id=author_id,
        units=(
            TranslationUnit(
                order=0,
                source_text="Hello",
                translated_text=translated_text,
                kind="paragraph",
                confidence=None,
            ),
        ),
    )


def test_version_numbers_are_assigned_by_the_database(session_factory):
    SqlResourceRepository(session_factory=session_factory).add(make_resource())
    versions = SqlVersionRepository(session_factory=session_factory)

    versions.save_version(make_version("v-1"))
    versions.save_version(
        make_version("v-2", author_kind=VersionAuthorKind.HUMAN, author_id="rev-1")
    )

    assert [v.version_number for v in versions.list_versions("res-1")] == [1, 2]
    assert versions.get_latest("res-1").author_id == "rev-1"


def test_the_machine_version_stays_retrievable_after_a_human_edit(session_factory):
    """Version 1 must never be overwritten.

    The MT-vs-human diff is both the audit trail and the training signal for
    the feedback loop, so this is the property the append-only design exists to
    provide.
    """
    SqlResourceRepository(session_factory=session_factory).add(make_resource())
    versions = SqlVersionRepository(session_factory=session_factory)
    versions.save_version(make_version("v-1", translated_text="Jambo"))
    versions.save_version(
        make_version(
            "v-2",
            author_kind=VersionAuthorKind.HUMAN,
            author_id="rev-1",
            translated_text="Habari za asubuhi",
        )
    )

    machine = versions.get_machine_version("res-1")

    assert machine is not None
    assert machine.version_number == 1
    assert machine.units[0].translated_text == "Jambo"
    assert versions.get_latest("res-1").units[0].translated_text == "Habari za asubuhi"


def test_save_version_if_current_appends_when_the_base_matches(session_factory):
    SqlResourceRepository(session_factory=session_factory).add(make_resource())
    versions = SqlVersionRepository(session_factory=session_factory)
    versions.save_version(make_version("v-1"))

    saved = versions.save_version_if_current(
        make_version(
            "v-2", author_kind=VersionAuthorKind.HUMAN, author_id="rev-1"
        ),
        base_version_number=1,
    )

    assert saved.version_number == 2
    assert len(versions.list_versions("res-1")) == 2


def test_save_version_if_current_refuses_a_stale_base(session_factory):
    SqlResourceRepository(session_factory=session_factory).add(make_resource())
    versions = SqlVersionRepository(session_factory=session_factory)
    versions.save_version(make_version("v-1"))
    versions.save_version(make_version("v-2"))

    with pytest.raises(StaleVersionError) as caught:
        versions.save_version_if_current(make_version("v-3"), base_version_number=1)

    assert caught.value.base_version_number == 1
    assert caught.value.current_version_number == 2
    assert len(versions.list_versions("res-1")) == 2


# ---------------------------------------------------------------------------
# SqlReviewRepository
# ---------------------------------------------------------------------------


def test_claim_next_takes_the_highest_priority_first(session_factory):
    SqlResourceRepository(session_factory=session_factory).add(make_resource())
    reviews = SqlReviewRepository(session_factory=session_factory)
    reviews.create_assignment(
        ReviewAssignment(
            assignment_id="low",
            resource_id="res-1",
            reviewer_id=None,
            assigned_at=NOW,
            priority=1,
        )
    )
    reviews.create_assignment(
        ReviewAssignment(
            assignment_id="high",
            resource_id="res-1",
            reviewer_id=None,
            assigned_at=NOW,
            priority=9,
        )
    )

    assert reviews.claim_next("rev-1").assignment_id == "high"
    assert reviews.claim_next("rev-2").assignment_id == "low"
    assert reviews.claim_next("rev-3") is None


def test_two_reviewers_never_claim_the_same_assignment(threaded_session_factory):
    """`claim_next()` must be atomic, for the same reason `save()` must be CAS.

    Reviewer time is the scarcest resource in this system. Two reviewers
    believing they own the same document wastes it twice.
    """
    SqlResourceRepository(session_factory=threaded_session_factory).add(make_resource())
    SqlReviewRepository(session_factory=threaded_session_factory).create_assignment(
        ReviewAssignment(
            assignment_id="only",
            resource_id="res-1",
            reviewer_id=None,
            assigned_at=NOW,
            priority=5,
        )
    )

    def claim(reviewer_id: str) -> str | None:
        claimed = SqlReviewRepository(
            session_factory=threaded_session_factory
        ).claim_next(reviewer_id)
        return claimed.assignment_id if claimed is not None else None

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = [
            future.result()
            for future in [pool.submit(claim, f"rev-{i}") for i in range(2)]
        ]

    assert sorted(c for c in claims if c is not None) == ["only"]


def test_a_reviewer_cannot_complete_someone_elses_assignment(session_factory):
    SqlResourceRepository(session_factory=session_factory).add(make_resource())
    reviews = SqlReviewRepository(session_factory=session_factory)
    reviews.create_assignment(
        ReviewAssignment(
            assignment_id="assign-1",
            resource_id="res-1",
            reviewer_id="rev-a",
            assigned_at=NOW,
        )
    )

    with pytest.raises(PermanentError, match="belongs to"):
        reviews.save_assignment(
            ReviewAssignment(
                assignment_id="assign-1",
                resource_id="res-1",
                reviewer_id="rev-b",
                decision=ReviewDecision.APPROVE,
                assigned_at=NOW,
            )
        )


def test_get_missing_assignment_raises_permanent_error(session_factory):
    reviews = SqlReviewRepository(session_factory=session_factory)

    with pytest.raises(PermanentError, match="not found"):
        reviews.get_assignment("does-not-exist")


def test_audit_events_round_trip_with_their_enums(session_factory):
    SqlResourceRepository(session_factory=session_factory).add(make_resource())
    reviews = SqlReviewRepository(session_factory=session_factory)

    reviews.append_audit(
        AuditEvent(
            event_id="evt-1",
            resource_id="res-1",
            actor_id="system:ingest",
            action="transition",
            from_status=ResourceStatus.SUBMITTED,
            to_status=ResourceStatus.FETCHED,
            at=NOW,
            details={"bytes": 1234},
        )
    )

    history = reviews.list_audit("res-1")

    assert len(history) == 1
    assert history[0].action == "transition"
    assert history[0].from_status is ResourceStatus.SUBMITTED
    assert history[0].to_status is ResourceStatus.FETCHED
    assert history[0].details == {"bytes": 1234}


def test_audit_events_come_back_oldest_first(session_factory):
    """Ordering is what makes the trail reconstructable."""
    SqlResourceRepository(session_factory=session_factory).add(make_resource())
    reviews = SqlReviewRepository(session_factory=session_factory)

    for index in range(3):
        reviews.append_audit(
            AuditEvent(
                event_id=f"evt-{index}",
                resource_id="res-1",
                actor_id="system:test",
                action="transition",
                from_status=None,
                to_status=ResourceStatus.FETCHED,
                at=NOW.replace(second=index),
                details={},
            )
        )

    assert [e.event_id for e in reviews.list_audit("res-1")] == [
        "evt-0",
        "evt-1",
        "evt-2",
    ]
