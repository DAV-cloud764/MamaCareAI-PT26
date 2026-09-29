"""Metrics tests PIPE-31: Null/timed/wiring, no prometheus_client needed."""

from __future__ import annotations

import uuid

import pytest
from backend.modules.pipeline.container import build_test_container
from backend.modules.pipeline.domain.enums import ResourceStatus, SourceType
from backend.modules.pipeline.domain.errors import PermanentError, TransientError
from backend.modules.pipeline.domain.models import Job, Resource
from backend.modules.pipeline.observability.metrics import (
    MetricNames,
    Metrics,
    NullMetrics,
    PrometheusMetrics,
)
from backend.modules.pipeline.stages.base import Stage, StageResult


class RecordingMetrics(Metrics):
    """In-memory sink asserting what stages emit."""

    def __init__(self) -> None:
        self.increments: list[tuple[str, dict[str, str]]] = []
        self.observations: list[tuple[str, float, dict[str, str]]] = []
        self.gauges: list[tuple[str, float, dict[str, str]]] = []

    def increment(self, name: str, *, labels: dict[str, str] | None = None) -> None:
        self.increments.append((name, dict(labels or {})))

    def observe(
        self, name: str, value: float, *, labels: dict[str, str] | None = None
    ) -> None:
        self.observations.append((name, value, dict(labels or {})))

    def gauge(
        self, name: str, value: float, *, labels: dict[str, str] | None = None
    ) -> None:
        self.gauges.append((name, value, dict(labels or {})))


class _ProbeStage(Stage):
    """Minimal stage: SUBMITTED in, FETCHED out (same arrow as ingest)."""

    def __init__(self, *, behaviour: str = "ok", **kwargs) -> None:
        super().__init__(**kwargs)
        self.behaviour = behaviour
        self.handled = 0

    @property
    def name(self) -> str:
        return "probe"

    @property
    def accepts(self) -> frozenset[ResourceStatus]:
        return frozenset({ResourceStatus.SUBMITTED})

    def handle(self, resource: Resource) -> StageResult:
        self.handled += 1
        if self.behaviour == "permanent":
            raise PermanentError("bad input, retrying changes nothing")
        if self.behaviour == "transient":
            raise TransientError("provider blip")
        return StageResult(next_status=ResourceStatus.FETCHED, next_stage=None)


def _seed_submitted(container) -> Resource:
    resource = Resource(
        resource_id=str(uuid.uuid4()),
        source_type=SourceType.WEB,
        source_url="https://example.org/metrics-probe",
        status=ResourceStatus.SUBMITTED,
    )
    container.resources.add(resource)
    return resource


def _run_probe(container, resource: Resource, behaviour: str, metrics: Metrics) -> _ProbeStage:
    stage = _ProbeStage(
        resources=container.resources,
        queue=container.queue,
        reviews=container.reviews,
        metrics=metrics,
        behaviour=behaviour,
    )
    stage.run(Job(job_id=str(uuid.uuid4()), resource_id=resource.resource_id, stage="probe"))
    return stage


def test_null_metrics_never_raise():
    metrics = NullMetrics()
    metrics.increment("anything", labels={"stage": "x"})
    metrics.observe("anything", 1.5, labels={"stage": "x"})
    metrics.gauge("anything", 3.0, labels={"stage": "x"})
    with metrics.timed("anything", labels={"stage": "x"}):
        pass
    with pytest.raises(RuntimeError), metrics.timed("anything"):
        raise RuntimeError("timed must not swallow the error")


def test_timed_records_duration_even_on_failure():
    rec = RecordingMetrics()
    with pytest.raises(ValueError), rec.timed("dur", labels={"stage": "probe"}):
        raise ValueError("boom")
    assert len(rec.observations) == 1
    name, value, labels = rec.observations[0]
    assert name == "dur"
    assert value >= 0
    assert labels == {"stage": "probe"}


def test_metric_names_unique():
    names = [
        MetricNames.JOBS_PROCESSED,
        MetricNames.STAGE_DURATION,
        MetricNames.QUEUE_DEPTH,
        MetricNames.DEAD_LETTER_DEPTH,
        MetricNames.REVIEW_QUEUE_AGE,
        MetricNames.TRANSLATION_CONFIDENCE,
    ]
    assert len(set(names)) == len(names)


def test_stage_emits_success_and_duration():
    container = build_test_container()
    resource = _seed_submitted(container)
    rec = RecordingMetrics()
    _run_probe(container, resource, "ok", rec)
    assert (MetricNames.JOBS_PROCESSED, {"stage": "probe", "outcome": "success"}) in rec.increments
    durations = [o for o in rec.observations if o[0] == MetricNames.STAGE_DURATION]
    assert len(durations) == 1
    assert durations[0][1] >= 0
    assert durations[0][2] == {"stage": "probe"}


def test_stage_emits_dead_letter_on_permanent_error():
    container = build_test_container()
    resource = _seed_submitted(container)
    rec = RecordingMetrics()
    _run_probe(container, resource, "permanent", rec)
    assert (
        MetricNames.JOBS_PROCESSED,
        {"stage": "probe", "outcome": "dead_letter"},
    ) in rec.increments
    # Duration is still recorded for the failed run.
    assert any(o[0] == MetricNames.STAGE_DURATION for o in rec.observations)


def test_stage_emits_retry_on_transient_error():
    container = build_test_container()
    resource = _seed_submitted(container)
    rec = RecordingMetrics()
    _run_probe(container, resource, "transient", rec)
    assert (MetricNames.JOBS_PROCESSED, {"stage": "probe", "outcome": "retry"}) in rec.increments


def test_stage_emits_skipped_for_wrong_status():
    container = build_test_container()
    resource = Resource(
        resource_id=str(uuid.uuid4()),
        source_type=SourceType.WEB,
        source_url="https://example.org/already-done",
        status=ResourceStatus.PUBLISHED,
    )
    container.resources.add(resource)
    rec = RecordingMetrics()
    stage = _run_probe(container, resource, "ok", rec)
    assert stage.handled == 0
    assert (
        MetricNames.JOBS_PROCESSED,
        {"stage": "probe", "outcome": "skipped"},
    ) in rec.increments


def test_stage_defaults_to_null_metrics():
    container = build_test_container()
    resource = _seed_submitted(container)
    stage = _ProbeStage(
        resources=container.resources,
        queue=container.queue,
        reviews=container.reviews,
    )
    assert isinstance(stage._metrics, NullMetrics)
    stage.run(Job(job_id=str(uuid.uuid4()), resource_id=resource.resource_id, stage="probe"))


def test_prometheus_metrics_safe_without_library():
    metrics = PrometheusMetrics()
    assert isinstance(metrics.available, bool)
    # Must never raise, with or without prometheus-client installed.
    metrics.increment(MetricNames.JOBS_PROCESSED, labels={"stage": "probe", "outcome": "success"})
    metrics.observe(MetricNames.STAGE_DURATION, 0.12, labels={"stage": "probe"})
    metrics.gauge(MetricNames.QUEUE_DEPTH, 4.0, labels={"stage": "probe"})
    with metrics.timed(MetricNames.STAGE_DURATION, labels={"stage": "probe"}):
        pass


def test_buckets_cover_slow_jobs_and_confidence_range():
    assert list(PrometheusMetrics.DURATION_BUCKETS) == sorted(
        PrometheusMetrics.DURATION_BUCKETS
    )
    assert PrometheusMetrics.DURATION_BUCKETS[-1] > 60
    assert min(PrometheusMetrics.CONFIDENCE_BUCKETS) >= 0
    assert max(PrometheusMetrics.CONFIDENCE_BUCKETS) <= 1.0
