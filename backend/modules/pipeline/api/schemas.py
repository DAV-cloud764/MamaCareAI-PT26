"""
API request/response schemas (Pydantic).

WHY THESE ARE SEPARATE FROM domain/models.py
It is tempting to return domain objects directly. Don't — they are different
things with different reasons to change:

  - Domain models change when the PIPELINE's rules change.
  - API schemas change when a CLIENT's needs change.

Coupling them means a harmless internal rename becomes a breaking API change,
and an internal field you never meant to expose (last_error with a stack trace,
an object storage key, a reviewer's id) gets serialized to whoever is calling.
The mapping is a few lines of boilerplate and it is worth it.

TODO (junior dev): implement these as pydantic BaseModel classes with real
validation. Validate at the EDGE so nothing invalid reaches a service.
"""

from __future__ import annotations

import ipaddress
import socket

from pydantic import BaseModel, Field, HttpUrl, field_validator

from ..domain.enums import ResourceStatus, ReviewDecision, SourceType


class SubmitRequest(BaseModel):
    source_url: HttpUrl
    source_type: SourceType | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("source_url")
    @classmethod
    def validate_public_url(cls, value: HttpUrl) -> HttpUrl:
        hostname = value.host

        if not hostname:
            raise ValueError("source URL must contain a hostname")

        try:
            addresses = socket.getaddrinfo(
                hostname,
                None,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise ValueError(
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
                raise ValueError(
                    "source URL must not resolve to a private or local address"
                )

        return value


class SubmitResponse(BaseModel):
    resource_id: str
    status: ResourceStatus


class ResourceStatusResponse(BaseModel):
    resource_id: str
    source_url: HttpUrl | None
    source_type: SourceType
    status: ResourceStatus
    detected_language: str | None = None
    language_confidence: float | None = None
    submitted_at: object
    updated_at: object
    current_version: int | None = None
    error: str | None = None


class PipelineStatsResponse(BaseModel):
    queue_depth: dict[str, int]
    resource_counts: dict[str, int]


class BlockSchema(BaseModel):
    order: int
    kind: str
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None


class UnitSchema(BaseModel):
    order: int
    source_text: str
    translated_text: str
    kind: str = "paragraph"
    confidence: float | None = None


class ReviewPayloadResponse(BaseModel):
    resource_id: str
    title: str | None
    source_language: str | None
    source_blocks: list[BlockSchema]
    translated_units: list[UnitSchema]
    machine_units: list[UnitSchema] | None
    version_number: int
    engine: str | None


class EditRequest(BaseModel):
    units: list[UnitSchema]
    base_version_number: int = Field(ge=1)
    note: str | None = None

    @field_validator("units")
    @classmethod
    def validate_units(cls, units: list[UnitSchema]) -> list[UnitSchema]:
        if not units:
            raise ValueError("an edit must contain at least one unit")

        orders = [unit.order for unit in units]
        if len(set(orders)) != len(orders):
            duplicates = sorted({o for o in orders if orders.count(o) > 1})
            raise ValueError(f"duplicate unit order(s): {duplicates}")
        if orders != sorted(orders):
            raise ValueError("units must be in ascending order")

        return units


class DecisionRequest(BaseModel):
    decision: ReviewDecision
    note: str | None = None

# ---------------------------------------------------------------------------
# TODO: implement the following schemas.
#
# --- Submission ---
#
# SubmitRequest
#     source_url: HttpUrl        <- HttpUrl, not str. Pydantic rejects
#                                   malformed and non-http URLs for free.
#     source_type: SourceType | None = None   (inferred when omitted)
#     metadata: dict[str, str] = {}
#     TODO: add a validator rejecting private/loopback hosts (SSRF guard —
#           see services/submission.py). Do it here AND in the service: here
#           for a good error message, there because the service is also
#           reachable from the CLI.
#
# SubmitResponse
#     resource_id: str
#     status: ResourceStatus
#     TODO: return 202 ACCEPTED, not 200. The work has been queued, not done.
#           The status code should tell the truth about what happened.
#
# --- Status ---
#
# ResourceStatusResponse
#     resource_id, source_url, source_type, status
#     detected_language, language_confidence
#     submitted_at, updated_at
#     current_version: int | None
#     error: str | None
#     TODO: NEVER put a raw exception traceback in `error`. Return a short,
#           safe message; the traceback belongs in the logs. Tracebacks leak
#           internal paths and library versions to whoever can call this.
#
# --- Review (PDF 3.6) ---
#
# ReviewPayloadResponse        <- powers the side-by-side view
#     resource_id, title, source_language
#     source_blocks: list[BlockSchema]        LEFT pane
#     translated_units: list[UnitSchema]      RIGHT pane, aligned by `order`
#     machine_units: list[UnitSchema] | None  for diffing after a human edit
#     version_number, engine
#
# EditRequest
#     units: list[UnitSchema]    (order + edited translated_text)
#     base_version_number: int   the version the reviewer was looking at
#     note: str | None
#     DONE: intrinsic validation — non-empty, unique orders, ascending.
#     NOT DONE, and deliberately not here: comparing the submitted orders
#     against the STORED version's orders, and checking base_version_number
#     against current state. A Pydantic model has no database access, so it
#     cannot know what the stored version looks like. That check belongs in
#     `VersionRepository.save_version_if_current` (raises StaleVersionError)
#     called from `ReviewService.submit_edit`, which the API maps to 409.
#     Splitting it this way is the point: the schema rejects a malformed
#     request, the repository rejects a stale one.
#
# DecisionRequest
#     decision: ReviewDecision
#     note: str | None
#     TODO: require a note when decision is NEEDS_EDIT or REJECT. "Rejected,
#           no reason given" is useless to the person who has to act on it, and
#           useless in the audit trail six months later.
# ---------------------------------------------------------------------------
