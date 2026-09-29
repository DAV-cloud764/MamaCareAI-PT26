
from __future__ import annotations

import uuid
from dataclasses import replace
from difflib import SequenceMatcher

from ..domain.enums import (
    JobStatus,
    ResourceStatus,
    ReviewDecision,
    VersionAuthorKind,
)
from ..domain.models import (
    AuditEvent,
    ContentVersion,
    Job,
    Resource,
    ReviewAssignment,
    TranslationUnit,
    utc_now,
)
from ..domain.state_machine import assert_can_transition
from ..ports.job_queue import JobQueue
from ..ports.repositories import (
    DocumentRepository,
    ResourceRepository,
    ReviewRepository,
    VersionRepository,
)
from ..ports.search_index import IndexedResource, SearchIndex


class ReviewService:
    """Drives the review state machine in response to human actions."""

    def __init__(
        self,
        *,
        resources: ResourceRepository,
        reviews: ReviewRepository,
        versions: VersionRepository,
        documents: DocumentRepository,
        queue: JobQueue,
        search: SearchIndex | None = None,
    ) -> None:
        self._resources = resources
        self._reviews = reviews
        self._versions = versions
        self._documents = documents
        self._queue = queue
        self._search = search

    def enqueue_for_review(
        self, resource: Resource, version: ContentVersion | None
    ) -> ReviewAssignment:
        """Create or return the open review task for a stored resource."""
        assignment_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"mamacare:review:{resource.resource_id}")
        )

        existing = self._get_assignment_if_present(assignment_id)
        if existing is not None and existing.completed_at is None:
            return existing

        assignment = ReviewAssignment(
            assignment_id=assignment_id,
            resource_id=resource.resource_id,
            reviewer_id=None,
            assigned_at=resource.submitted_at,
            priority=self._calculate_priority(version),
        )
        try:
            self._reviews.create_assignment(assignment)
        except Exception:
            concurrent = self._get_assignment_if_present(assignment_id)
            if concurrent is not None and concurrent.completed_at is None:
                return concurrent
            raise
        return assignment

    def claim_next(self, reviewer_id: str) -> ReviewAssignment | None:
        """Give a reviewer their next piece of work."""
        self._require_reviewer_id(reviewer_id)
        assignment = self._reviews.claim_next(reviewer_id)
        if assignment is None:
            return None

        resource = self._resources.get(assignment.resource_id)
        if resource.status == ResourceStatus.STORED:
            assert_can_transition(resource.status, ResourceStatus.IN_REVIEW)
            updated = resource.with_status(ResourceStatus.IN_REVIEW)
            self._resources.save(updated, expected_status=resource.status)
            self._append_audit(
                resource=resource,
                reviewer_id=reviewer_id,
                action="claim",
                target=ResourceStatus.IN_REVIEW,
                details={"assignment_id": assignment.assignment_id},
            )
        elif resource.status != ResourceStatus.IN_REVIEW:
            assert_can_transition(resource.status, ResourceStatus.IN_REVIEW)

        return assignment

    def get_review_payload(self, resource_id: str) -> dict[str, object]:
        """Return the source, current translation, and version history."""
        resource = self._resources.get(resource_id)
        document = self._documents.get_document(resource_id)
        versions = sorted(
            self._versions.list_versions(resource_id),
            key=lambda item: item.version_number,
        )
        latest = versions[-1] if versions else None
        source_blocks = tuple(sorted(document.blocks, key=lambda block: block.order))
        latest_units = (
            tuple(sorted(latest.units, key=lambda unit: unit.order)) if latest else ()
        )

        machine_version = None
        if latest is not None and latest.author_kind == VersionAuthorKind.HUMAN:
            machine_version = self._versions.get_machine_version(resource_id)

        blocks_by_order = {block.order: block for block in source_blocks}
        units_by_order = {unit.order: unit for unit in latest_units}
        aligned_content = tuple(
            {
                "order": order,
                "source": blocks_by_order.get(order),
                "translation": units_by_order.get(order),
            }
            for order in sorted(blocks_by_order.keys() | units_by_order.keys())
        )
        version_history = tuple(
            {
                "version_id": version.version_id,
                "version_number": version.version_number,
                "author_kind": version.author_kind,
                "author_id": version.author_id,
                "created_at": version.created_at,
                "engine": version.engine,
                "note": version.note,
            }
            for version in versions
        )

        return {
            "resource": resource,
            "document": document,
            "source_blocks": source_blocks,
            "latest_version": latest,
            "latest_units": latest_units,
            "translation_units": latest_units,
            "machine_version": machine_version,
            "version_history": version_history,
            "aligned_content": aligned_content,
        }

    def confirm_language(
        self, *, resource_id: str, reviewer_id: str, language: str
    ) -> None:
        """Record a reviewer's language confirmation and resume processing."""
        self._require_reviewer_id(reviewer_id)
        resource = self._resources.get(resource_id)
        if resource.status is not ResourceStatus.NEEDS_LANGUAGE_CONFIRMATION:
            raise ValueError("Resource is not awaiting language confirmation")

        updated = resource.with_status(
            resource.status,
            detected_language=language,
            language_confidence=1.0,
            source_metadata={
                **resource.source_metadata,
                "language_confirmed_by": reviewer_id,
            },
        )
        self._resources.save(updated, expected_status=resource.status)
        self._reviews.append_audit(
            AuditEvent(
                event_id=str(uuid.uuid4()),
                resource_id=resource_id,
                actor_id=reviewer_id,
                action="confirm_language",
                from_status=resource.status,
                to_status=resource.status,
                details={"language": language},
            )
        )
        self._queue.publish(
            Job(
                job_id=str(uuid.uuid4()),
                resource_id=resource_id,
                stage="detect_language",
                status=JobStatus.PENDING,
            )
        )

    def submit_edit(
        self,
        *,
        assignment_id: str,
        reviewer_id: str,
        edited_units: list[TranslationUnit],
        note: str | None = None,
    ) -> ContentVersion:
        """Save a reviewer's corrections as a new version.

        The resource transition is committed BEFORE the version is appended,
        and that order is load-bearing. Doing it the other way round means a
        losing concurrent edit has already written its version by the time the
        conditional update refuses it, leaving an orphaned version that no
        status ever points at and no reviewer can see. Since versions are
        append-only, that orphan is permanent.

        The two writes are still not one transaction, so the reverse is
        possible: the transition lands and the version write then fails,
        leaving the resource in EDITED with nothing new to review. That failure
        is visible — EDITED has no outgoing transition, so the next attempt
        fails the state machine loudly — whereas a silent orphan is not. A
        single unit of work spanning both repositories is the proper fix and
        belongs with the persistence factory.
        """
        self._require_reviewer_id(reviewer_id)
        assignment = self._owned_assignment(assignment_id, reviewer_id)
        resource = self._resources.get(assignment.resource_id)
        assert_can_transition(resource.status, ResourceStatus.EDITED)

        # Claim the transition first. If a concurrent edit got here first, this
        # raises and nothing has been written.
        self._resources.save(
            resource.with_status(ResourceStatus.EDITED),
            expected_status=resource.status,
        )

        previous = self._versions.get_latest(resource.resource_id)
        next_number = 1 if previous is None else previous.version_number + 1
        candidate = ContentVersion(
            version_id=str(uuid.uuid4()),
            resource_id=resource.resource_id,
            version_number=next_number,
            author_kind=VersionAuthorKind.HUMAN,
            author_id=reviewer_id,
            units=tuple(sorted(edited_units, key=lambda unit: unit.order)),
            note=note,
        )
        self._versions.save_version(candidate)

        saved = next(
            (
                version
                for version in self._versions.list_versions(resource.resource_id)
                if version.version_id == candidate.version_id
            ),
            None,
        )
        if saved is None:
            raise RuntimeError(
                f"Version repository did not persist edit {candidate.version_id}"
            )

        machine = self._versions.get_machine_version(resource.resource_id)
        differences = self._calculate_differences(machine, saved)
        self._append_audit(
            resource=resource,
            reviewer_id=reviewer_id,
            action="edit",
            target=ResourceStatus.EDITED,
            details={
                "assignment_id": assignment_id,
                "version_id": saved.version_id,
                "version_number": saved.version_number,
                "note": note,
                "differences": differences,
            },
        )
        self._reindex(
            self._resources.get(resource.resource_id), saved
        )
        return saved

    def submit_decision(
        self,
        *,
        assignment_id: str,
        reviewer_id: str,
        decision: ReviewDecision,
        note: str | None = None,
    ) -> None:
        """Record a review decision and move the workflow forward."""
        self._require_reviewer_id(reviewer_id)
        assignment = self._owned_assignment(assignment_id, reviewer_id)
        decision = ReviewDecision(decision)

        if decision in {ReviewDecision.NEEDS_EDIT, ReviewDecision.REJECT} and not (
            note and note.strip()
        ):
            raise ValueError(
                f"A note is required when the decision is {decision.value}"
            )

        resource = self._resources.get(assignment.resource_id)
        target = {
            ReviewDecision.APPROVE: ResourceStatus.APPROVED,
            ReviewDecision.NEEDS_EDIT: ResourceStatus.NEEDS_EDIT,
            ReviewDecision.REJECT: ResourceStatus.FAILED,
        }[decision]
        assert_can_transition(resource.status, target)

        changes: dict[str, object] = {}
        if decision == ReviewDecision.APPROVE:
            changes["source_metadata"] = {
                **resource.source_metadata,
                "approved_by": reviewer_id,
            }
        elif decision == ReviewDecision.REJECT:
            changes["last_error"] = note

        updated = resource.with_status(target, **changes)
        self._resources.save(updated, expected_status=resource.status)

        completed_at = None
        if decision in {ReviewDecision.APPROVE, ReviewDecision.REJECT}:
            completed_at = utc_now()
        self._reviews.save_assignment(
            replace(assignment, decision=decision, completed_at=completed_at)
        )
        self._append_audit(
            resource=resource,
            reviewer_id=reviewer_id,
            action=decision.value,
            target=target,
            details={"assignment_id": assignment_id, "note": note},
        )

        if decision == ReviewDecision.APPROVE:
            self._queue.publish(
                Job(
                    job_id=str(uuid.uuid4()),
                    resource_id=resource.resource_id,
                    stage="publish",
                    status=JobStatus.PENDING,
                )
            )

    @staticmethod
    def _calculate_priority(version: ContentVersion | None) -> int:
        """Turn mean confidence into a sortable integer (higher is sooner)."""
        if version is None:
            mean_confidence = 0.0
        else:
            confidences = [
                unit.confidence for unit in version.units if unit.confidence is not None
            ]
            mean_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )
        bounded = min(1.0, max(0.0, mean_confidence))
        return round((1.0 - bounded) * 1000)

    def _get_assignment_if_present(self, assignment_id: str) -> ReviewAssignment | None:
        try:
            return self._reviews.get_assignment(assignment_id)
        except (KeyError, LookupError):
            return None
        except Exception as exc:
            if exc.__class__.__name__ == "AssignmentNotFoundError":
                return None
            raise

    def _owned_assignment(
        self, assignment_id: str, reviewer_id: str
    ) -> ReviewAssignment:
        assignment = self._reviews.get_assignment(assignment_id)
        if assignment.reviewer_id != reviewer_id:
            raise PermissionError(
                f"Assignment {assignment_id} is not owned by reviewer {reviewer_id}"
            )
        if assignment.completed_at is not None:
            raise ValueError(f"Assignment {assignment_id} is already complete")
        return assignment

    @staticmethod
    def _require_reviewer_id(reviewer_id: str) -> None:
        if not reviewer_id or not reviewer_id.strip():
            raise ValueError("reviewer_id is required")

    def _append_audit(
        self,
        *,
        resource: Resource,
        reviewer_id: str,
        action: str,
        target: ResourceStatus,
        details: dict[str, object],
    ) -> None:
        self._reviews.append_audit(
            AuditEvent(
                event_id=str(uuid.uuid4()),
                resource_id=resource.resource_id,
                actor_id=reviewer_id,
                action=action,
                from_status=resource.status,
                to_status=target,
                details=details,
            )
        )

    @staticmethod
    def _calculate_differences(
        machine: ContentVersion | None, human: ContentVersion
    ) -> list[dict[str, object]]:
        if machine is None:
            return []
        machine_by_order = {unit.order: unit for unit in machine.units}
        differences: list[dict[str, object]] = []
        for unit in human.units:
            original = machine_by_order.get(unit.order)
            machine_text = original.translated_text if original else ""
            ratio = SequenceMatcher(None, machine_text, unit.translated_text).ratio()
            differences.append(
                {
                    "order": unit.order,
                    "machine_translation": machine_text,
                    "human_translation": unit.translated_text,
                    "edit_distance": 1.0 - ratio,
                }
            )
        return differences

    def _reindex(self, resource: Resource, version: ContentVersion) -> None:
        if self._search is None:
            return
        self._search.index(
            IndexedResource(
                resource_id=resource.resource_id,
                title=resource.source_metadata.get("title"), # type: ignore
                translated_text="\n\n".join(
                    unit.translated_text for unit in version.units
                ),
                source_url=resource.source_url,
                status=resource.status.value,
                version_number=version.version_number,
                metadata={"language": resource.detected_language or ""},
            )
        )