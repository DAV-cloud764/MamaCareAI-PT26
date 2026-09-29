"""HTTP endpoints used by the human-review application."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from ..container import Container
from ..domain.enums import ResourceStatus, ReviewDecision, VersionAuthorKind
from ..domain.errors import InvalidStateTransition, StaleVersionError
from ..domain.models import (
    ContentVersion,
    NormalizedDocument,
    Resource,
    TextBlock,
    TranslationUnit,
)
from ..services.review_service import ReviewService
from .schemas import (
    BlockSchema,
    DecisionRequest,
    EditRequest,
    ReviewPayloadResponse,
    UnitSchema,
)

router = APIRouter(prefix="/review", tags=["review"])


class ReviewUser(BaseModel):
    """The authenticated identity supplied by the application's auth layer."""

    user_id: str
    roles: set[str] = Field(default_factory=set)


class VersionSummary(BaseModel):
    version_number: int
    author_kind: VersionAuthorKind
    author_id: str | None
    created_at: datetime
    engine: str | None
    note: str | None


class AuditEventSchema(BaseModel):
    event_id: str
    resource_id: str
    actor_id: str
    action: str
    from_status: ResourceStatus | None
    to_status: ResourceStatus | None
    at: datetime
    details: dict[str, object]


class LanguageConfirmationRequest(BaseModel):
    language: str = Field(min_length=2, max_length=8, pattern=r"^[A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?$")


def get_container(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("Pipeline container is not initialized")
    return container


def get_review_service(
    container: Annotated[Container, Depends(get_container)],
) -> ReviewService:
    return ReviewService(
        resources=container.resources,
        reviews=container.reviews,
        versions=container.versions,
        documents=container.documents,
        queue=container.queue,
    )


def get_current_user(request: Request) -> ReviewUser:
    """Read the identity installed by authentication middleware."""
    principal: Any = getattr(request.state, "user", None)
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if isinstance(principal, ReviewUser):
        return principal
    if isinstance(principal, dict):
        user_id = principal.get("user_id") or principal.get("id")
        if not isinstance(user_id, str) or not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication identity")
        return ReviewUser(
            user_id=user_id,
            roles=set(principal.get("roles", ())),
        )
    user_id = getattr(principal, "user_id", getattr(principal, "id", None))
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication identity")
    return ReviewUser(
        user_id=user_id,
        roles=set(getattr(principal, "roles", ())),
    )


def _require_role(user: ReviewUser, *allowed: str) -> None:
    if not user.roles.intersection(allowed):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")


def _payload(service: ReviewService, resource_id: str) -> ReviewPayloadResponse:
    payload = service.get_review_payload(resource_id)
    resource = cast(Resource, payload["resource"])
    document = cast(NormalizedDocument, payload["document"])
    latest = cast(ContentVersion | None, payload["latest_version"])
    machine = cast(ContentVersion | None, payload["machine_version"])
    if latest is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Resource has no translation version to review",
        )
    return ReviewPayloadResponse(
        resource_id=resource.resource_id,
        title=document.title,
        source_language=resource.detected_language,
        source_blocks=[
            BlockSchema.model_validate(block, from_attributes=True)
            for block in cast(tuple[TextBlock, ...], payload["source_blocks"])
        ],
        translated_units=[
            UnitSchema.model_validate(unit, from_attributes=True)
            for unit in cast(tuple[TranslationUnit, ...], payload["latest_units"])
        ],
        machine_units=(
            [
                UnitSchema.model_validate(unit, from_attributes=True)
                for unit in machine.units
            ]
            if machine is not None
            else None
        ),
        version_number=latest.version_number,
        engine=latest.engine,
    )


@router.get("/next", response_model=ReviewPayloadResponse)
def claim_next(
    response: Response,
    user: Annotated[ReviewUser, Depends(get_current_user)],
    service: Annotated[ReviewService, Depends(get_review_service)],
) -> ReviewPayloadResponse | None:
    _require_role(user, "reviewer", "approver")
    assignment = service.claim_next(user.user_id)
    if assignment is None:
        response.status_code = status.HTTP_204_NO_CONTENT
        return None
    return _payload(service, assignment.resource_id)


@router.get("/{resource_id}/versions", response_model=list[VersionSummary])
def list_versions(
    resource_id: str,
    user: Annotated[ReviewUser, Depends(get_current_user)],
    container: Annotated[Container, Depends(get_container)],
) -> list[VersionSummary]:
    _require_role(user, "read_only", "reviewer", "approver")
    return [
        VersionSummary(
            version_number=item.version_number,
            author_kind=item.author_kind,
            author_id=item.author_id,
            created_at=item.created_at,
            engine=item.engine,
            note=item.note,
        )
        for item in container.versions.list_versions(resource_id)
    ]


@router.get("/{resource_id}/audit", response_model=list[AuditEventSchema])
def list_audit(
    resource_id: str,
    user: Annotated[ReviewUser, Depends(get_current_user)],
    container: Annotated[Container, Depends(get_container)],
) -> list[AuditEventSchema]:
    _require_role(user, "read_only", "reviewer", "approver")
    return [
        AuditEventSchema.model_validate(event, from_attributes=True)
        for event in container.reviews.list_audit(resource_id)
    ]


@router.post("/{assignment_id}/edit", status_code=status.HTTP_201_CREATED)
def submit_edit(
    assignment_id: str,
    payload: EditRequest,
    user: Annotated[ReviewUser, Depends(get_current_user)],
    service: Annotated[ReviewService, Depends(get_review_service)],
) -> dict[str, int]:
    _require_role(user, "reviewer", "approver")
    units = [
        TranslationUnit(
            order=unit.order,
            source_text=unit.source_text,
            translated_text=unit.translated_text,
            kind=unit.kind,
            confidence=unit.confidence,
        )
        for unit in payload.units
    ]
    try:
        version = service.submit_edit(
            assignment_id=assignment_id,
            reviewer_id=user.user_id,
            edited_units=units,
            base_version_number=payload.base_version_number,
            note=payload.note,
        )
    except StaleVersionError as exc:
        # 409, not 500. The reviewer did nothing wrong — their tab went stale
        # while they were typing, which is the normal case for two reviewers on
        # one document. Tell them how far behind they are so the UI can offer a
        # reload, rather than showing a generic failure.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": str(exc),
                "base_version_number": exc.base_version_number,
                "current_version_number": exc.current_version_number,
            },
        ) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except InvalidStateTransition as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return {"version_number": version.version_number}


@router.post("/{assignment_id}/decision", status_code=status.HTTP_200_OK)
def submit_decision(
    assignment_id: str,
    payload: DecisionRequest,
    user: Annotated[ReviewUser, Depends(get_current_user)],
    service: Annotated[ReviewService, Depends(get_review_service)],
) -> dict[str, str]:
    _require_role(user, "reviewer", "approver")
    if payload.decision is ReviewDecision.APPROVE:
        _require_role(user, "approver")
    if payload.decision in {ReviewDecision.NEEDS_EDIT, ReviewDecision.REJECT} and not (payload.note or "").strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A note is required for this decision")
    try:
        service.submit_decision(
            assignment_id=assignment_id,
            reviewer_id=user.user_id,
            decision=payload.decision,
            note=payload.note,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    return {"decision": payload.decision.value}


@router.post("/{resource_id}/language", status_code=status.HTTP_200_OK)
def confirm_language(
    resource_id: str,
    payload: LanguageConfirmationRequest,
    user: Annotated[ReviewUser, Depends(get_current_user)],
    service: Annotated[ReviewService, Depends(get_review_service)],
) -> dict[str, str]:
    _require_role(user, "reviewer", "approver")
    service.confirm_language(
        resource_id=resource_id,
        reviewer_id=user.user_id,
        language=payload.language.lower(),
    )
    return {"language": payload.language.lower()}


@router.get("/{resource_id}", response_model=ReviewPayloadResponse)
def get_review_payload(
    resource_id: str,
    user: Annotated[ReviewUser, Depends(get_current_user)],
    service: Annotated[ReviewService, Depends(get_review_service)],
) -> ReviewPayloadResponse:
    _require_role(user, "read_only", "reviewer", "approver")
    return _payload(service, resource_id)
