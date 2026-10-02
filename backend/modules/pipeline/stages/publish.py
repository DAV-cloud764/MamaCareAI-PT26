"""
Stage 7: Published output (PDF 3.7) + the compliance gate (PDF section 4).

Approved, versioned Swahili content becomes searchable and available to
downstream consumers.

THE COMPLIANCE GATE IS PART OF THIS STAGE, NOT AN EARLIER ONE.
The design doc is explicit: "add a licensing/compliance gate before
publication, not only before translation." Reason: publication is the act with
legal consequence. Translating something we may not republish costs us a few
cents; publishing it is the actual problem. So the last thing that happens
before content goes live is a licensing check.

FOR MAMACARE AI SPECIFICALLY: published output is the input to
`backend/modules/knowledge` (chunking + embedding into the vector store). This
stage is the seam between "vetted content" and "what the bot is allowed to say"
— which makes it the enforcement point for ARCHITECTURE.md non-negotiable #4:
every source in the knowledge base is traceable to a vetted register entry.
"""

from __future__ import annotations

import uuid

from ..domain.enums import ResourceStatus
from ..domain.errors import InvalidStateTransition, TransientError
from ..domain.models import AuditEvent, ContentVersion, Resource
from ..observability.metrics import Metrics
from ..ports.job_queue import JobQueue
from ..ports.repositories import ResourceRepository, ReviewRepository, VersionRepository
from ..ports.search_index import IndexedResource, SearchIndex
from ..services import ComplianceGate
from .base import Stage, StageResult


class KnowledgeHandoffError(TransientError):
    """The knowledge module refused the published content.

    Retriable on purpose. A handoff is a call to another service, and "that
    service is briefly down" is exactly the case `TransientError` exists for —
    `Stage.run` will re-queue with backoff. It used to subclass bare
    `Exception`, which `base.py` treats as a bug and dead-letters on the first
    failure, so one unavailable dependency permanently lost an approved
    document.
    """


class PublishStage(Stage):
    """Runs the compliance gate, then publishes approved content."""

    def __init__(
        self,
        *,
        resources: ResourceRepository,
        queue: JobQueue,
        reviews: ReviewRepository,
        versions: VersionRepository,
        search: SearchIndex,
        compliance_gate: ComplianceGate,
        knowledge_handoff: object | None = None,  # TODO: type as knowledge.KnowledgeHandoff
        max_attempts: int = 5,
        metrics: Metrics | None = None,
    ) -> None:
        super().__init__(
            resources=resources, queue=queue, reviews=reviews, max_attempts=max_attempts, metrics=metrics
        )
        self._versions = versions
        self._reviews = reviews
        self._search = search
        self._compliance = compliance_gate
        self._knowledge_handoff = knowledge_handoff

    @property
    def name(self) -> str:
        return "publish"

    @property
    def accepts(self) -> frozenset[ResourceStatus]:
        return frozenset({ResourceStatus.APPROVED})

    def handle(self, resource: Resource) -> StageResult:
        """Check licensing, hand off, then publish the approved version.

        Order is non-negotiable, and the handoff now comes BEFORE the index
        write. It used to come after, while its own comment said it should not
        fail the publish — so a handoff error raised *after* the index had
        already been mutated, leaving the resource in APPROVED with a PUBLISHED
        index entry. Those two disagreed about the same document and nothing
        reconciled them.

        The version published is the one approval named, not the newest one.
        `submit_decision` records `approved_version_id` at the moment of the
        click; publishing `get_latest()` instead would be correct only for as
        long as nothing else was appended afterwards.
        """
        decision = self._compliance.evaluate(resource)
        if not decision.allowed:
            self._audit(
                resource,
                action="publish_blocked",
                from_status=ResourceStatus.APPROVED,
                to_status=ResourceStatus.BLOCKED_LICENSING,
                details={"reason": decision.reason, "license_id": decision.license_id},
            )
            return StageResult(
                next_status=ResourceStatus.BLOCKED_LICENSING,
                next_stage=None,
                details={"reason": decision.reason},
            )

        version = self._approved_version(resource)

        translated_text = "\n\n".join(unit.translated_text for unit in version.units)
        # `source_metadata` is a free-form bag, so this needs narrowing rather
        # than trusting whatever the fetcher happened to put there.
        raw_title = resource.source_metadata.get("title")
        title = raw_title if isinstance(raw_title, str) else None

        if self._knowledge_handoff is not None:
            try:
                self._knowledge_handoff.handoff_published_content(
                    resource_id=resource.resource_id,
                    source_url=resource.source_url,
                    title=title,
                    translated_text=translated_text,
                    version_number=version.version_number,
                    language=resource.detected_language or "",
                    metadata=resource.source_metadata,
                )
            except Exception as exc:
                # Raised BEFORE the index write, so a failed handoff leaves
                # nothing published anywhere. `Stage.run` re-queues it.
                raise KnowledgeHandoffError(
                    f"Failed to hand off resource {resource.resource_id} to the "
                    f"knowledge module: {exc}",
                    resource_id=resource.resource_id,
                ) from exc

        self._search.index(
            IndexedResource(
                resource_id=resource.resource_id,
                title=title,
                translated_text=translated_text,
                source_url=resource.source_url,
                status=ResourceStatus.PUBLISHED.value,
                version_number=version.version_number,
                metadata={"language": resource.detected_language or ""},
            )
        )

        self._audit(
            resource,
            action="publish",
            from_status=ResourceStatus.APPROVED,
            to_status=ResourceStatus.PUBLISHED,
            details={
                "approved_version": version.version_number,
                "approved_by": resource.source_metadata.get("approved_by"),
                "knowledge_handoff": self._knowledge_handoff is not None,
            },
        )

        return StageResult(
            next_status=ResourceStatus.PUBLISHED,
            next_stage=None,
            details={
                "approved_version": version.version_number,
                "approved_by": resource.source_metadata.get("approved_by"),
                "knowledge_handoff": self._knowledge_handoff is not None,
            },
        )

    def _approved_version(self, resource: Resource) -> ContentVersion:
        """Resolve the version approval named, refusing if there is none.

        Falling back to the newest version would reintroduce the bug this
        replaces: "whatever happens to be latest" is not the same claim as
        "a human approved this".
        """
        approved_id = resource.source_metadata.get("approved_version_id")
        if isinstance(approved_id, str) and approved_id:
            for version in self._versions.list_versions(resource.resource_id):
                if version.version_id == approved_id:
                    return version
            raise InvalidStateTransition(
                f"Resource {resource.resource_id} was approved at version "
                f"{approved_id!r}, which is no longer retrievable.",
                resource_id=resource.resource_id,
            )

        raise InvalidStateTransition(
            f"Resource {resource.resource_id} is APPROVED but records no "
            f"approved version, so there is nothing authorised to publish.",
            resource_id=resource.resource_id,
        )

    def _audit(
        self,
        resource: Resource,
        *,
        action: str,
        from_status: ResourceStatus,
        to_status: ResourceStatus,
        details: dict[str, object],
    ) -> None:
        """Append the audit row.

        The module's own non-negotiable is that every action writes an audit
        event, and `ComplianceGate` documents that every evaluation — pass or
        fail — must be recorded by the calling stage. Publication was doing
        neither.
        """
        self._reviews.append_audit(
            AuditEvent(
                event_id=str(uuid.uuid4()),
                resource_id=resource.resource_id,
                actor_id=f"system:{self.name}",
                action=action,
                from_status=from_status,
                to_status=to_status,
                details=details,
            )
        )
