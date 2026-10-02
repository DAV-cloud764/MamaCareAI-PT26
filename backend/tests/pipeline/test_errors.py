"""
Domain error taxonomy tests.

`stages/base.py` reads exactly one thing off an exception to decide what
happens next: `retryable`. True re-queues the job with backoff; False sends it
straight to the dead-letter queue. So the classification is not documentation,
it is control flow, and a subclass declared in the wrong place changes what
the pipeline does when a stage fails.

These tests pin the classification, not the messages.
"""

from __future__ import annotations

from backend.modules.pipeline.domain import errors


def test_stale_version_is_permanent() -> None:
    """Retrying a stale write loses the same edit, so it must not be retried.

    The reviewer has to reload, see what changed, and decide again. Re-running
    the identical write would either collide again or, worse, be treated as a
    transient blip and retried until the resource is dead-lettered.
    """
    assert issubclass(errors.StaleVersionError, errors.PermanentError)
    assert errors.StaleVersionError.retryable is False


def test_stale_version_carries_both_numbers() -> None:
    """The API needs to tell the reviewer how far behind they are.

    The audit trail needs the same pair, otherwise the collision is recorded as
    a bare refusal with nothing to reconstruct it from.
    """
    error = errors.StaleVersionError(
        "stale",
        resource_id="r1",
        base_version_number=2,
        current_version_number=3,
    )

    assert error.resource_id == "r1"
    assert error.base_version_number == 2
    assert error.current_version_number == 3


def test_permanent_errors_are_not_retryable() -> None:
    for error_type in (
        errors.UnsupportedSourceType,
        errors.ExtractionError,
        errors.LanguageDetectionUncertain,
        errors.TranslationError,
        errors.InvalidStateTransition,
        errors.ComplianceBlocked,
        errors.StaleVersionError,
    ):
        assert error_type.retryable is False, error_type.__name__


def test_transient_errors_are_retryable() -> None:
    for error_type in (errors.TransientError, errors.FetchError, errors.ProviderRateLimited):
        assert error_type.retryable is True, error_type.__name__


def test_every_error_descends_from_pipeline_error() -> None:
    for error_type in (
        errors.TransientError,
        errors.PermanentError,
        errors.FetchError,
        errors.ProviderRateLimited,
        errors.UnsupportedSourceType,
        errors.ExtractionError,
        errors.LanguageDetectionUncertain,
        errors.TranslationError,
        errors.InvalidStateTransition,
        errors.ComplianceBlocked,
        errors.StaleVersionError,
    ):
        assert issubclass(error_type, errors.PipelineError), error_type.__name__
