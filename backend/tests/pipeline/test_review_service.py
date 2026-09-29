"""Executable specification for the human-review workflow (PDF 3.6)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from threading import Barrier, Lock
from typing import cast

import pytest
from backend.modules.pipeline.domain.enums import (
    ResourceStatus,
    ReviewDecision,
    SourceType,
    VersionAuthorKind,
)
from backend.modules.pipeline.domain.models import (
    ContentVersion,
    NormalizedDocument,
    Resource,
    ReviewAssignment,
    TextBlock,
    TranslationUnit,
)
from backend.modules.pipeline.services.review_service import ReviewService
from backend.tests.pipeline.fakes import (
    FakeDocumentRepository,
    FakeJobQueue,
    FakeResourceRepository,
    FakeReviewRepository,
    FakeVersionRepository,
)


class ReviewRepo(FakeReviewRepository):
    """Faithful in-memory implementation of atomic, priority-ordered claiming."""

    def __init__(self) -> None:
        super().__init__()
        self.lock = Lock()

    def claim_next(self, reviewer_id: str) -> ReviewAssignment | None:
        with self.lock:
            open_items = [a for a in self._assignments.values() if a.reviewer_id is None and a.completed_at is None]
            if not open_items:
                return None
            item = min(open_items, key=lambda a: (-a.priority, a.assigned_at))
            item = replace(item, reviewer_id=reviewer_id)
            self._assignments[item.assignment_id] = item
            return item


class VersionRepo(FakeVersionRepository):
    def __init__(self) -> None:
        super().__init__()
        self.lock = Lock()

    def save_version(self, version: ContentVersion) -> None:
        with self.lock:
            super().save_version(version)


class ConcurrentReadResourceRepository(FakeResourceRepository):
    """Make both edit requests observe the same valid pre-edit state."""

    def __init__(self) -> None:
        super().__init__()
        self.barrier = Barrier(2)
        self.reads_left = 2
        self.read_lock = Lock()

    def get(self, resource_id: str) -> Resource:
        item = super().get(resource_id)
        with self.read_lock:
            should_wait = self.reads_left > 0
            if should_wait:
                self.reads_left -= 1
        if should_wait:
            self.barrier.wait()
        return item


def _unit(order: int, text: str, confidence: float = 0.8) -> TranslationUnit:
    return TranslationUnit(order=order, source_text=f"source-{order}", translated_text=text, confidence=confidence)


def _resource(resource_id: str, status: ResourceStatus) -> Resource:
    return Resource(resource_id=resource_id, source_type=SourceType.WEB, source_url=f"https://example.test/{resource_id}", status=status, detected_language="en")


def _machine(resource_id: str, confidence: float = 0.8) -> ContentVersion:
    return ContentVersion(version_id=f"{resource_id}-v1", resource_id=resource_id, version_number=1, author_kind=VersionAuthorKind.MACHINE, author_id=None, engine="test-mt", units=(_unit(0, "machine-0", confidence), _unit(1, "machine-1", confidence)))


def _assignment(resource_id: str, reviewer_id: str | None = "reviewer-1", **changes: object) -> ReviewAssignment:
    values = {"assignment_id": f"assignment-{resource_id}", "resource_id": resource_id, "reviewer_id": reviewer_id}
    values.update(changes)
    return ReviewAssignment(**values)  # type: ignore[arg-type]


def _build(status: ResourceStatus = ResourceStatus.NEEDS_EDIT, reviewer_id: str | None = "reviewer-1"):
    resources, reviews, versions = FakeResourceRepository(), ReviewRepo(), VersionRepo()
    documents, queue = FakeDocumentRepository(), FakeJobQueue()
    resources.add(_resource("r1", status))
    versions.save_version(_machine("r1"))
    review = _assignment("r1", reviewer_id)
    reviews.create_assignment(review)
    service = ReviewService(resources=resources, reviews=reviews, versions=versions, documents=documents, queue=queue)
    return service, resources, reviews, versions, documents, queue, review


def _edits(prefix: str = "human") -> list[TranslationUnit]:
    return [_unit(0, f"{prefix}-0"), _unit(1, f"{prefix}-1")]


def test_edit_creates_a_new_version_and_preserves_the_machine_output() -> None:
    service, _, _, versions, _, _, review = _build()
    machine_version = versions.get_machine_version("r1")
    assert machine_version is not None
    original = asdict(machine_version)
    created = service.submit_edit(assignment_id=review.assignment_id, reviewer_id="reviewer-1", edited_units=_edits())
    history = versions.list_versions("r1")
    assert created.version_number == 2
    assert [v.version_number for v in history] == [1, 2]
    assert asdict(history[0]) == original


def test_second_edit_creates_version_3() -> None:
    service, _, _, versions, _, _, review = _build()
    service.submit_edit(assignment_id=review.assignment_id, reviewer_id="reviewer-1", edited_units=_edits("first"))
    service.submit_decision(assignment_id=review.assignment_id, reviewer_id="reviewer-1", decision=ReviewDecision.NEEDS_EDIT, note="More changes")
    created = service.submit_edit(assignment_id=review.assignment_id, reviewer_id="reviewer-1", edited_units=_edits("second"))
    assert created.version_number == 3
    assert [v.version_number for v in versions.list_versions("r1")] == [1, 2, 3]


def test_concurrent_edits_do_not_collide_on_version_number() -> None:
    resources = ConcurrentReadResourceRepository()
    reviews, versions = ReviewRepo(), VersionRepo()
    resources.add(_resource("r1", ResourceStatus.NEEDS_EDIT))
    versions.save_version(_machine("r1"))
    first = _assignment("r1")
    reviews.create_assignment(first)
    second = _assignment("r1", "reviewer-2", assignment_id="assignment-2")
    reviews.create_assignment(second)
    service = ReviewService(
        resources=resources,
        reviews=reviews,
        versions=versions,
        documents=FakeDocumentRepository(),
        queue=FakeJobQueue(),
    )

    def submit(item: ReviewAssignment) -> ContentVersion:
        assert item.reviewer_id
        return service.submit_edit(assignment_id=item.assignment_id, reviewer_id=item.reviewer_id, edited_units=_edits(item.reviewer_id))

    with ThreadPoolExecutor(max_workers=2) as pool:
        created = list(pool.map(submit, (first, second)))
    assert len({v.version_number for v in created}) == 2
    assert [v.version_number for v in versions.list_versions("r1")] == [1, 2, 3]


def test_claim_next_returns_highest_priority_first() -> None:
    resources, reviews, versions = FakeResourceRepository(), ReviewRepo(), VersionRepo()
    now = datetime.now(UTC)
    for rid, confidence, priority, age in (("safe", .95, 50, 2), ("risky", .2, 800, 1)):
        resources.add(_resource(rid, ResourceStatus.STORED))
        versions.save_version(_machine(rid, confidence))
        reviews.create_assignment(_assignment(rid, None, priority=priority, assigned_at=now - timedelta(hours=age)))
    service = ReviewService(resources=resources, reviews=reviews, versions=versions, documents=FakeDocumentRepository(), queue=FakeJobQueue())
    assert service.claim_next("reviewer-1").resource_id == "risky"  # type: ignore[union-attr]


def test_two_reviewers_never_claim_the_same_assignment() -> None:
    service, _, _, _, _, _, review = _build(ResourceStatus.STORED, None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(service.claim_next, ("reviewer-1", "reviewer-2")))
    assert [a.assignment_id for a in claims if a] == [review.assignment_id]


def test_claim_next_on_an_empty_queue_returns_none() -> None:
    service = ReviewService(resources=FakeResourceRepository(), reviews=ReviewRepo(), versions=VersionRepo(), documents=FakeDocumentRepository(), queue=FakeJobQueue())
    assert service.claim_next("reviewer-1") is None


def test_approve_transitions_to_approved_and_queues_publication() -> None:
    service, resources, reviews, _, _, queue, review = _build(ResourceStatus.IN_REVIEW)
    service.submit_decision(assignment_id=review.assignment_id, reviewer_id="reviewer-1", decision=ReviewDecision.APPROVE)
    assert resources.get("r1").status is ResourceStatus.APPROVED
    assert reviews.get_assignment(review.assignment_id).completed_at is not None
    job = queue.claim_next("publish")
    assert job is not None and job.resource_id == "r1"


def test_needs_edit_keeps_the_assignment_open() -> None:
    service, resources, reviews, _, _, _, review = _build(ResourceStatus.IN_REVIEW)
    service.submit_decision(assignment_id=review.assignment_id, reviewer_id="reviewer-1", decision=ReviewDecision.NEEDS_EDIT, note="Fix wording")
    saved = reviews.get_assignment(review.assignment_id)
    assert resources.get("r1").status is ResourceStatus.NEEDS_EDIT
    assert saved.decision is ReviewDecision.NEEDS_EDIT and saved.completed_at is None


def test_reject_records_the_reason() -> None:
    service, resources, reviews, _, _, _, review = _build(ResourceStatus.IN_REVIEW)
    reason = "Source and translation discuss different medicines"
    service.submit_decision(assignment_id=review.assignment_id, reviewer_id="reviewer-1", decision=ReviewDecision.REJECT, note=reason)
    assert resources.get("r1").status is ResourceStatus.FAILED
    assert reviews.list_audit("r1")[-1].details["note"] == reason


def test_reviewer_cannot_complete_someone_elses_assignment() -> None:
    service, resources, reviews, _, _, _, review = _build(ResourceStatus.IN_REVIEW)
    with pytest.raises(PermissionError):
        service.submit_decision(assignment_id=review.assignment_id, reviewer_id="intruder", decision=ReviewDecision.APPROVE)
    assert resources.get("r1").status is ResourceStatus.IN_REVIEW
    assert reviews.get_assignment(review.assignment_id).completed_at is None


@pytest.mark.parametrize("action", ["claim", "edit", "approve"])
def test_every_action_writes_an_audit_event(action: str) -> None:
    status = {"claim": ResourceStatus.STORED, "edit": ResourceStatus.NEEDS_EDIT, "approve": ResourceStatus.IN_REVIEW}[action]
    service, _, reviews, _, _, _, review = _build(status, None if action == "claim" else "reviewer-1")
    if action == "claim":
        service.claim_next("reviewer-1")
    elif action == "edit":
        service.submit_edit(assignment_id=review.assignment_id, reviewer_id="reviewer-1", edited_units=_edits())
    else:
        service.submit_decision(assignment_id=review.assignment_id, reviewer_id="reviewer-1", decision=ReviewDecision.APPROVE)
    events = reviews.list_audit("r1")
    assert len(events) == 1
    assert (events[0].action, events[0].actor_id) == (action, "reviewer-1")


def test_audit_trail_is_ordered_and_complete() -> None:
    service, _, reviews, _, _, _, review = _build(ResourceStatus.STORED, None)
    service.claim_next("reviewer-1")
    service.submit_decision(assignment_id=review.assignment_id, reviewer_id="reviewer-1", decision=ReviewDecision.NEEDS_EDIT, note="Correct it")
    service.submit_edit(assignment_id=review.assignment_id, reviewer_id="reviewer-1", edited_units=_edits())
    service.submit_decision(assignment_id=review.assignment_id, reviewer_id="reviewer-1", decision=ReviewDecision.APPROVE)
    events = reviews.list_audit("r1")
    assert [e.action for e in events] == ["claim", "needs_edit", "edit", "approve"]
    assert [e.actor_id for e in events] == ["reviewer-1"] * 4
    assert [e.at for e in events] == sorted(e.at for e in events)


def _save_document(documents: FakeDocumentRepository) -> None:
    documents.save_document(NormalizedDocument(resource_id="r1", title="Care", author=None, published_date=None, blocks=(TextBlock(order=1, kind="paragraph", text="source-1"), TextBlock(order=0, kind="paragraph", text="source-0"))))


def test_payload_aligns_source_and_translation_by_order() -> None:
    service, _, _, _, documents, _, _ = _build()
    _save_document(documents)
    payload = service.get_review_payload("r1")
    source_blocks = cast(tuple[TextBlock, ...], payload["source_blocks"])
    translation_units = cast(tuple[TranslationUnit, ...], payload["translation_units"])
    source_orders = [block.order for block in source_blocks]
    translation_orders = [item.order for item in translation_units]
    assert source_orders == translation_orders == [0, 1]


def test_payload_includes_the_machine_version_after_a_human_edit() -> None:
    service, _, _, _, documents, _, review = _build()
    _save_document(documents)
    service.submit_edit(assignment_id=review.assignment_id, reviewer_id="reviewer-1", edited_units=_edits())
    payload = service.get_review_payload("r1")
    translation_units = cast(tuple[TranslationUnit, ...], payload["translation_units"])
    assert [u.translated_text for u in translation_units] == ["human-0", "human-1"]
    machine = cast(ContentVersion | None, payload["machine_version"])
    assert machine is not None
    assert [u.translated_text for u in machine.units] == ["machine-0", "machine-1"]
