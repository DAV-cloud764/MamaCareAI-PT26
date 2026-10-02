"""
Stage base class — retries, idempotency, transitions, and metrics in ONE place.

WHY A BASE CLASS HERE (and not copy-paste in seven files)
Every stage needs identical plumbing: load the resource, check it is in a state
this stage handles, do the work, transition it, publish the next job, record
the audit event, emit metrics, and decide retry-vs-dead-letter on failure.
Written seven times, that plumbing is wrong in at least three of them within a
month — and the wrong ones will be the stages nobody looks at.

So `run()` is a TEMPLATE METHOD: it owns the plumbing and calls the one
abstract hook each stage actually implements. Subclasses write business logic
only. When you need to change retry policy, you change it here, once.

This is also Open/Closed at the stage level: new stage = new subclass, no edits
to the runner or the worker.

IMPLEMENTATION NOTES:
- This implements the template method pattern
- Every stage inherits from this and only needs to implement the abstract methods
- The run() method handles all the plumbing (loading, validation, persistence, error handling)
- Requires Abdillah's technical lead review before merging
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from abc import ABC, abstractmethod

from backend.modules.pipeline.domain.enums import JobStatus, ResourceStatus
from backend.modules.pipeline.domain.errors import (
    PipelineError,
    ProviderRateLimited,
)
from backend.modules.pipeline.domain.models import AuditEvent, Job, Resource

# Use the actual state machine from repository (it's now implemented)
from backend.modules.pipeline.domain.state_machine import assert_can_transition
from backend.modules.pipeline.observability.metrics import (
    MetricNames,
    Metrics,
    NullMetrics,
)
from backend.modules.pipeline.ports.job_queue import JobQueue
from backend.modules.pipeline.ports.repositories import (
    ResourceRepository,
    ReviewRepository,
)

logger = logging.getLogger(__name__)


class StageResult:
    """What a stage's business logic reports back to the template method.

    Returning a result object instead of mutating shared state keeps the
    "what happened" decision in the stage and the "how it is persisted"
    decision in the base class.

    Attributes:
        next_status: state to move the resource into.
        next_stage: stage to enqueue afterwards; None ends this path
            (e.g. the resource now waits for a human).
        resource_changes: extra fields to persist on the Resource
            (detected_language, raw_object_key, ...).
        details: anything worth putting in the audit event.
    """

    def __init__(
        self,
        *,
        next_status: ResourceStatus,
        next_stage: str | None = None,
        resource_changes: dict[str, object] | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        self.next_status = next_status
        self.next_stage = next_stage
        self.resource_changes = resource_changes or {}
        self.details = details or {}


class Stage(ABC):
    """Template for every pipeline stage."""

    def __init__(
        self,
        *,
        resources: ResourceRepository,
        queue: JobQueue,
        reviews: ReviewRepository,
        max_attempts: int = 5,
        metrics: Metrics | None = None,
    ) -> None:
        # Dependencies arrive here, always. A stage that builds its own S3
        # client in a method cannot be tested without AWS credentials.
        # Metrics default to a no-op so existing constructions and every
        # test keep working without a metrics server.
        self._resources = resources
        self._queue = queue
        self._reviews = reviews
        self._max_attempts = max_attempts
        self._metrics = metrics if metrics is not None else NullMetrics()

    # --- What each subclass must declare -------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name used as the queue topic and in metrics, e.g. "extract"."""

    @property
    @abstractmethod
    def accepts(self) -> frozenset[ResourceStatus]:
        """Statuses this stage is willing to process.

        The guard against out-of-order and duplicate work. If a redelivered job
        arrives for a resource that has already moved on, `run()` sees the
        status is not in `accepts` and skips it as a no-op instead of
        re-processing. This is how at-least-once delivery is made safe.
        """

    @abstractmethod
    def handle(self, resource: Resource) -> StageResult:
        """The actual work of this stage. THE ONLY METHOD A SUBCLASS WRITES.

        Preconditions guaranteed by `run()`:
          - `resource` was just loaded fresh from the repository.
          - `resource.status` is in `self.accepts`.

        Rules:
          - Raise a `PipelineError` subclass on failure; `run()` decides retry
            vs dead-letter from its `retryable` flag.
          - Do not call `self._resources.save()` — return a StageResult and let
            `run()` persist. One writer means one place where transitions can
            go wrong.
          - Be idempotent. Assume this may be the second time you have run.
        """
        raise NotImplementedError

    # --- The plumbing every stage shares -------------------------------------

    def run(self, job: Job) -> None:
        """Execute one job end to end. Implemented ONCE, inherited by all stages.

        IMPLEMENTATION NOTES:
        1. LOAD fresh state - never trust job payload
        2. IDEMPOTENCY GUARD - skip if status not in accepts
        3. DO THE WORK - call handle() with timing
        4. VALIDATE TRANSITION - use state machine
        5. PERSIST - conditional update
        6. AUDIT - log the action
        7. ENQUEUE NEXT STAGE - only after commit
        8. ERROR HANDLING - retry with backoff or dead-letter
        9. EMIT METRICS - success/failure counters
        """
        start_time = time.time()

        try:
            # 1. LOAD fresh state
            resource = self._resources.get(job.resource_id)

            # 2. IDEMPOTENCY GUARD
            if resource.status not in self.accepts:
                logger.info(
                    f"Stage {self.name}: resource {job.resource_id} status "
                    f"{resource.status} not in accepts, skipping (already processed or out of order)"
                )
                self._metrics.increment(
                    MetricNames.JOBS_PROCESSED,
                    labels={"stage": self.name, "outcome": "skipped"},
                )
                return

            # 3. DO THE WORK (timed even on failure — the failed duration
            # is the one you want when finding the broken stage)
            try:
                with self._metrics.timed(
                    MetricNames.STAGE_DURATION, labels={"stage": self.name}
                ):
                    result = self.handle(resource)
            except PipelineError as exc:
                # 8. ERROR HANDLING - PipelineError subclasses
                if exc.retryable and job.attempts < self._max_attempts:
                    # Retry with backoff
                    backoff = self._backoff_seconds(job.attempts)
                    not_before = time.time() + backoff

                    # Check if provider gave us a specific retry time
                    if isinstance(exc, ProviderRateLimited) and exc.retry_after_seconds:
                        not_before = time.time() + exc.retry_after_seconds

                    logger.warning(
                        f"Stage {self.name}: retryable error for {job.resource_id}, "
                        f"attempt {job.attempts + 1}/{self._max_attempts}, retry in {backoff:.1f}s"
                    )

                    # Re-queue with backoff
                    retry_job = Job(
                        job_id=job.job_id,
                        resource_id=job.resource_id,
                        stage=job.stage,
                        status=JobStatus.RETRYING,
                        attempts=job.attempts + 1,
                        enqueued_at=job.enqueued_at,
                        not_before=not_before,
                    )
                    self._queue.publish(retry_job)
                    self._metrics.increment(
                        MetricNames.JOBS_PROCESSED,
                        labels={"stage": self.name, "outcome": "retry"},
                    )
                    return
                else:
                    # Non-retryable or max attempts exceeded - dead letter
                    logger.error(
                        f"Stage {self.name}: non-retryable error or max attempts exceeded "
                        f"for {job.resource_id}, sending to dead letter: {exc}"
                    )
                    # Mark resource as failed
                    failed_resource = resource.with_status(
                        ResourceStatus.FAILED,
                        last_error=str(exc),
                        attempt_count=resource.attempt_count + 1,
                    )
                    self._resources.save(
                        failed_resource, expected_status=resource.status
                    )
                    self._queue.send_to_dead_letter(job, reason=str(exc))
                    self._metrics.increment(
                        MetricNames.JOBS_PROCESSED,
                        labels={"stage": self.name, "outcome": "dead_letter"},
                    )
                    return

            except Exception as exc:
                # Unexpected error - bug, not transient fault
                logger.exception(
                    f"Stage {self.name}: unexpected error for {job.resource_id}",
                )
                # Dead letter it - retrying a bug just multiplies noise
                failed_resource = resource.with_status(
                    ResourceStatus.FAILED,
                    last_error=f"Unexpected error: {exc}",
                    attempt_count=resource.attempt_count + 1,
                )
                self._resources.save(
                    failed_resource, expected_status=resource.status
                )
                self._queue.send_to_dead_letter(job, reason=f"Unexpected error: {exc}")
                self._metrics.increment(
                    MetricNames.JOBS_PROCESSED,
                    labels={"stage": self.name, "outcome": "dead_letter"},
                )
                return

            # 4. VALIDATE THE TRANSITION before writing
            assert_can_transition(resource.status, result.next_status)

            # 5. PERSIST
            # `expected_status` is the status we loaded, not the one we are
            # about to write. That is what makes the write conditional, and
            # therefore what makes a lost update detectable at all.
            updated = resource.with_status(
                result.next_status,
                attempt_count=resource.attempt_count + 1,
                **result.resource_changes,
            )
            self._resources.save(updated, expected_status=resource.status)

            # 6. AUDIT
            audit_event = AuditEvent(
                event_id=str(uuid.uuid4()),
                resource_id=resource.resource_id,
                actor_id=f"system:{self.name}",
                action="transition",
                from_status=resource.status,
                to_status=result.next_status,
                details=result.details,
            )
            self._reviews.append_audit(audit_event)

            # 7. ENQUEUE THE NEXT STAGE (only after the state change is committed)
            if result.next_stage:
                next_job = Job(
                    job_id=str(uuid.uuid4()),
                    resource_id=resource.resource_id,
                    stage=result.next_stage,
                    status=JobStatus.PENDING,
                    attempts=0,
                )
                self._queue.publish(next_job)

            # 9. EMIT SUCCESS METRICS
            duration = time.time() - start_time
            self._metrics.increment(
                MetricNames.JOBS_PROCESSED,
                labels={"stage": self.name, "outcome": "success"},
            )
            logger.info(
                f"Stage {self.name}: completed {job.resource_id} in {duration:.3f}s, "
                f"transition {resource.status} -> {result.next_status}"
            )

        except Exception as exc:
            # This should not happen if error handling above is correct
            logger.critical(
                f"Stage {self.name}: unhandled error in run() for {job.resource_id}: {exc}",
                exc_info=True,
            )
            raise

    def _backoff_seconds(self, attempt: int) -> float:
        """Exponential backoff with jitter.

        IMPLEMENTATION: min(base * 2 ** attempt, cap) * random_factor

        The jitter is not optional. Without it, 200 jobs that failed together
        during a provider outage all retry at the exact same instant and knock
        the provider over again the moment it recovers. This is the
        thundering-herd problem, and one line of random.uniform prevents it.
        """
        base = 1.0  # 1 second base
        cap = 300.0  # 5 minute cap

        # Exponential backoff
        backoff = min(base * (2 ** attempt), cap)

        # Add jitter: random factor in [0.5, 1.5]
        jitter = random.uniform(0.5, 1.5)

        return backoff * jitter
