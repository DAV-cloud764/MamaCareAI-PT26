"""
Pipeline exception hierarchy.

WHY THIS MATTERS FOR ROBUSTNESS
The orchestrator has to answer one question every time a stage throws:
"should I retry this, or is retrying pointless?" Retrying a 503 from a
translation API is correct. Retrying a password-protected PDF forever burns
money and hides the real problem in a retry loop.

So every exception here declares `retryable`. `stages/base.py` reads that flag
and either re-queues with backoff or sends the job straight to the
dead-letter queue.

TODO (junior dev):
  [ ] Never raise a bare `Exception` inside a stage. Wrap it in the right
      subclass here so the retry decision is explicit and reviewable.
  [ ] When you add an adapter, map its library-specific errors onto these
      types at the adapter boundary — stages must never catch `botocore` or
      `httpx` exceptions directly.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for every error raised inside the pipeline.

    Attributes:
        retryable: True if re-running the same job later could plausibly
            succeed (network blip, rate limit, provider outage). False if the
            input itself is the problem (corrupt file, unsupported format).
    """

    retryable: bool = False

    def __init__(self, message: str, *, resource_id: str | None = None) -> None:
        super().__init__(message)
        self.resource_id = resource_id


# --- Transient: retry with backoff -------------------------------------------------


class TransientError(PipelineError):
    """Something outside our control failed and may work on retry."""

    retryable = True


class FetchError(TransientError):
    """Source could not be downloaded (timeout, 5xx, DNS, connection reset)."""


class ProviderRateLimited(TransientError):
    """MT / ASR / OCR provider returned a rate-limit or quota response.

    TODO: adapters raising this SHOULD set `retry_after_seconds` when the
    provider tells us how long to wait — blind exponential backoff wastes time
    the provider already quantified for us.
    """

    def __init__(
        self,
        message: str,
        *,
        resource_id: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message, resource_id=resource_id)
        self.retry_after_seconds = retry_after_seconds


# --- Permanent: do not retry, route to dead-letter / human ------------------------


class PermanentError(PipelineError):
    """The input or configuration is wrong. Retrying changes nothing."""

    retryable = False


class UnsupportedSourceType(PermanentError):
    """No fetcher/extractor is registered for this source type."""


class ExtractionError(PermanentError):
    """File was downloaded but no usable text could be produced.

    Example: an encrypted PDF, a video with no audio track and no captions.
    These need a human decision, not a retry.
    """


class LanguageDetectionUncertain(PermanentError):
    """Detection confidence fell below the configured threshold.

    Not a crash — a routing signal. The stage catches this and moves the
    resource to NEEDS_LANGUAGE_CONFIRMATION so a human can confirm the
    language, because (PDF 3.3) bad detection silently poisons translation.
    """

    def __init__(
        self,
        message: str,
        *,
        resource_id: str | None = None,
        detected_language: str | None = None,
        confidence: float | None = None,
    ) -> None:
        super().__init__(message, resource_id=resource_id)
        self.detected_language = detected_language
        self.confidence = confidence


class TranslationError(PermanentError):
    """Translation failed in a way retrying will not fix (e.g. unsupported language pair)."""


class InvalidStateTransition(PermanentError):
    """A stage tried to move a resource into a state that is not reachable from its current one.

    This almost always means two workers processed the same resource
    concurrently, or a stage ran out of order. Treat it as a bug, not as noise.
    """


class ComplianceBlocked(PermanentError):
    """The licensing/compliance gate refused publication (PDF section 4).

    Terminal by design: content we do not have the right to republish must
    never reach the published index, no matter how good the translation is.
    """


class StaleVersionError(PermanentError):
    """A write was based on a version that is no longer the current one.

    Raised by `VersionRepository.save_version_if_current` when the caller's
    `base_version_number` is behind the stored state. Not retryable, because
    retrying the identical write loses the same edit: the reviewer has to
    reload, see what changed, and decide again.

    Carries both numbers so the API layer can tell the reviewer exactly how far
    behind they are, and so the audit trail records the collision rather than
    just the refusal.
    """

    def __init__(
        self,
        message: str,
        *,
        resource_id: str | None = None,
        base_version_number: int | None = None,
        current_version_number: int | None = None,
    ) -> None:
        super().__init__(message, resource_id=resource_id)
        self.base_version_number = base_version_number
        self.current_version_number = current_version_number
