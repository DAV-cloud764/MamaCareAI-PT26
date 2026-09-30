"""
Submission service — the front door of the pipeline.

Accepts a URL, validates it, creates the Resource record, and enqueues the
first job. Returns IMMEDIATELY with a resource_id.

WHY IT RETURNS IMMEDIATELY
"Every source type becomes a job on a queue, not a synchronous call." A
submission endpoint that waits for a 40MB PDF to download will time out, and
the user will click submit again, and now you have duplicate work plus an
angry user. Accept, enqueue, return an id they can poll.

MAMACARE-SPECIFIC GATE (ARCHITECTURE.md non-negotiable #4):
Nothing enters this pipeline that is not already vetted in
`data/01_source_register`. This service is where that is enforced. Do not add a
"just this once" bypass — an unvetted source that reaches the knowledge base
breaks the traceability guarantee the whole project rests on.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

from ..domain.enums import ResourceStatus, SourceType
from ..domain.errors import PermanentError, UnsupportedSourceType
from ..domain.models import Job, Resource
from ..ports.job_queue import JobQueue
from ..ports.repositories import ResourceRepository


class SubmissionService:
    """Creates resources and starts them down the pipeline."""

    def __init__(
        self,
        *,
        resources: ResourceRepository,
        queue: JobQueue,
        source_register: object,
    ) -> None:
        self._resources = resources
        self._queue = queue
        self._source_register = source_register

    @staticmethod
    def _normalize_url(source_url: str) -> str:
        """Return a canonical URL for duplicate checking and storage."""
        parsed = urlparse(source_url)

        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()

        netloc = hostname

        if parsed.username is not None:
            userinfo = parsed.username
            if parsed.password is not None:
                userinfo = f"{userinfo}:{parsed.password}"
            netloc = f"{userinfo}@{netloc}"

        if parsed.port is not None:
            is_default_port = (
                (scheme == "http" and parsed.port == 80)
                or (scheme == "https" and parsed.port == 443)
            )
            if not is_default_port:
                netloc = f"{netloc}:{parsed.port}"

        return urlunparse(
            (
                scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                "",
            )
        )

    @staticmethod
    def _validate_url(source_url: str) -> None:
        """Validate URL scheme and prevent SSRF against local/private hosts."""
        parsed = urlparse(source_url)

        if parsed.scheme.lower() not in {"http", "https"}:
            raise PermanentError("source URL must use http or https")

        hostname = parsed.hostname

        if not hostname:
            raise PermanentError("source URL must contain a hostname")

        try:
            addresses = socket.getaddrinfo(
                hostname,
                None,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise PermanentError(
                "source URL hostname could not be resolved"
            ) from exc

        seen: set[str] = set()

        for address in addresses:
            ip_text = address[4][0]

            if ip_text in seen:
                continue

            seen.add(ip_text)
            ip = ipaddress.ip_address(ip_text)

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_unspecified
            ):
                raise PermanentError(
                    "source URL resolves to a private or local address"
                )

    @staticmethod
    def _infer_source_type(source_url: str) -> SourceType:
        """Infer a source type from the URL when the caller did not provide one."""
        parsed = urlparse(source_url)
        hostname = (parsed.hostname or "").lower()
        path = parsed.path.lower()

        if path.endswith(".pdf"):
            return SourceType.PDF

        # Basic platform sniffing for video sources.
        video_hosts = {
            "youtube.com",
            "www.youtube.com",
            "youtu.be",
            "vimeo.com",
            "www.vimeo.com",
        }

        if hostname in video_hosts or hostname.endswith(".youtube.com"):
            return SourceType.VIDEO

        return SourceType.WEB

    def _check_duplicate(self, source_url: str) -> Resource | None:
        """
        Check for a resource already submitted for the same URL.

        The current repository contract exposes content-hash lookup rather than
        URL lookup, so we use the existing hash mechanism with an empty content
        value, matching the ingest-stage URL-level dedup strategy.
        """
        # Import locally so the dependency remains small and obvious.
        from ..ports.deduplicator import Deduplicator

        del Deduplicator  # type-only architectural reference

        import hashlib

        url_hash = hashlib.sha256(
            source_url.encode("utf-8")
        ).hexdigest()

        existing = self._resources.find_by_content_hash(url_hash)

        if existing is None:
            return None

        if existing.status in {
            ResourceStatus.SUBMITTED,
            ResourceStatus.FETCHED,
            ResourceStatus.EXTRACTED,
            ResourceStatus.LANGUAGE_DETECTED,
            ResourceStatus.NEEDS_LANGUAGE_CONFIRMATION,
            ResourceStatus.TRANSLATED,
            ResourceStatus.STORED,
            ResourceStatus.IN_REVIEW,
            ResourceStatus.NEEDS_EDIT,
            ResourceStatus.EDITED,
            ResourceStatus.APPROVED,
            ResourceStatus.PUBLISHED,
        }:
            return existing

        return None

    def submit(
        self,
        *,
        source_url: str,
        source_type: SourceType | None = None,
        submitted_by: str,
        metadata: dict[str, object] | None = None,
    ) -> Resource:
        """Accept a resource for processing and return its record."""

        source_url = self._normalize_url(source_url)

        # 1. Validate URL and SSRF constraints.
        self._validate_url(source_url)

        # 2. Enforce the MamaCare source-register gate.
        is_approved = self._source_register.is_approved(source_url)

        if not is_approved:
            raise PermanentError(
                "source not vetted; add it to data/01_source_register first"
            )

        # 3. Infer source type only when the caller did not provide one.
        resolved_source_type = (
            source_type
            if source_type is not None
            else self._infer_source_type(source_url)
        )

        if not isinstance(resolved_source_type, SourceType):
            raise UnsupportedSourceType(
                f"Unsupported source type: {resolved_source_type!r}"
            )

        # 4. Cheap deduplication check.
        existing = self._check_duplicate(source_url)

        if existing is not None:
            return existing

        # 5. Create the Resource.
        resource = Resource(
            resource_id=str(uuid4()),
            source_type=resolved_source_type,
            source_url=source_url,
            status=ResourceStatus.SUBMITTED,
            source_metadata={
                **(metadata or {}),
                "submitted_by": submitted_by,
            },
        )

        # 6. Persist BEFORE publishing the job.
        self._resources.add(resource)

        # 7. Enqueue the first stage.
        job = Job(
            job_id=str(uuid4()),
            resource_id=resource.resource_id,
            stage="ingest",
        )

        self._queue.publish(job)

        # 8. Return immediately.
        return resource
