"""
Pipeline admin API — submit resources, check status.

    POST /pipeline/resources           submit a URL
    POST /pipeline/resources/upload    submit a PDF file directly
    GET  /pipeline/resources/{id}      status + version history
    GET  /pipeline/resources           list by status (the ops view)
    GET  /pipeline/stats               queue depths, counts per status

NOT A PUBLIC SURFACE. This is an internal tool for the data team. It must sit
behind authentication before it is exposed anywhere — an unauthenticated
endpoint that fetches arbitrary URLs is an open proxy, and someone will find it.
"""


from __future__ import annotations

from hashlib import sha256
from typing import Annotated
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)

from ..adapters.storage.keys import build_raw_key
from ..container import Container
from ..domain.enums import ResourceStatus, SourceType
from ..domain.errors import PermanentError, ResourceNotFound
from ..domain.models import Job, Resource
from ..services.submission import SubmissionService
from .schemas import (
    PipelineStatsResponse,
    ResourceStatusResponse,
    SubmitRequest,
    SubmitResponse,
)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def get_container(request: Request) -> Container:
    """Return the application-wide dependency container."""
    container = getattr(request.app.state, "container", None)

    if container is None:
        raise RuntimeError("Pipeline container is not initialized")

    return container


def get_submission_service(
    container: Annotated[Container, Depends(get_container)],
) -> SubmissionService:
    """
    Build a lightweight service around the shared dependencies.

    The container itself remains application-scoped; no database/model
    container is rebuilt per request.
    """
    return SubmissionService(
        resources=container.resources,
        queue=container.queue,
        source_register=getattr(container, "source_register", None),
    )


@router.post(
    "/resources",
    response_model=SubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_resource(
    payload: SubmitRequest,
    service: Annotated[SubmissionService, Depends(get_submission_service)],
) -> SubmitResponse:
    """Submit a URL and enqueue ingestion immediately."""
    if service._source_register is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Source register is not configured",
        )

    try:
        resource = service.submit(
            source_url=str(payload.source_url),
            source_type=payload.source_type,
            submitted_by="system",
            metadata=payload.metadata,
        )
    except PermanentError as exc:
        message = str(exc)

        if "not vetted" in message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message,
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        ) from exc

    return SubmitResponse(
        resource_id=resource.resource_id,
        status=resource.status,
    )

@router.post(
    "/resources/upload",
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_resource(
    file: UploadFile,
    container: Annotated[Container, Depends(get_container)],
):
        if file.filename is None:
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A file is required",
        )

        header = await file.read(5)

        if header != b"%PDF-":
          raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

        await file.seek(0)

        max_size = container.settings.fetch_max_bytes
        total_size = 0

        chunks: list[bytes] = []

        while True:
           chunk = await file.read(1024 * 1024)

           if not chunk:
               break

           total_size += len(chunk)

           if total_size > max_size:
               raise HTTPException(
                   status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                   detail="Uploaded file exceeds the maximum allowed size",
               )

           chunks.append(chunk)

        content = b"".join(chunks)

        resource = Resource(
        resource_id=str(uuid4()),
        source_type=SourceType.PDF,
        source_url=None,
        status=ResourceStatus.FETCHED,
    )

        raw_key = build_raw_key(resource, extension="pdf")

        stored_key = container.object_store.put(
        raw_key,
        content,
        content_type="application/pdf",
    )

        resource = resource.with_status(
        ResourceStatus.FETCHED,
        raw_object_key=stored_key,
        content_hash=sha256(content).hexdigest(),
    )
        container.resources.add(resource)

        job = Job(
        job_id=str(uuid4()),
        resource_id=resource.resource_id,
        stage="extract",
    )

        container.queue.publish(job)

        return SubmitResponse(
        resource_id=resource.resource_id,
        status=resource.status,
    )


@router.get(
   "/stats",
   response_model=PipelineStatsResponse,
)
def get_stats(
   container: Annotated[Container, Depends(get_container)],
) -> PipelineStatsResponse:
   stages = ("ingest", "extract", "detect_language", "translate", "store", "review", "publish")
   queue_depth = {stage: container.queue.depth(stage) for stage in stages}

   page_size = 500
   resource_counts: dict[str, int] = {}
   for resource_status in ResourceStatus:
       total = 0
       offset = 0
       while True:
           items = container.resources.list_by_status(
               resource_status,
               limit=page_size,
               offset=offset,
           )
           if not items:
               break
           total += len(items)
           if len(items) < page_size:
               break
           offset += page_size
       resource_counts[resource_status.value] = total

   return PipelineStatsResponse(
       queue_depth=queue_depth,
       resource_counts=resource_counts,
   )


@router.get(
   "/resources",
   response_model=list[ResourceStatusResponse],
)
def list_resources(
    container: Annotated[Container, Depends(get_container)],
    status: Annotated[ResourceStatus, Query(...)],
    limit: Annotated[int, Query(ge=1)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ResourceStatusResponse]:
   effective_limit = min(limit, 500)
   resources = container.resources.list_by_status(
       status,
       limit=effective_limit,
       offset=offset,
   )

   return [
       ResourceStatusResponse(
           resource_id=resource.resource_id,
           source_url=resource.source_url,
           source_type=resource.source_type,
           status=resource.status,
           detected_language=resource.detected_language,
           language_confidence=resource.language_confidence,
           submitted_at=resource.submitted_at,
           updated_at=resource.updated_at,
           error=resource.last_error,
       )
       for resource in resources
   ]


@router.get(
   "/resources/{resource_id}",
   response_model=ResourceStatusResponse,
)
def get_resource_status(
    resource_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> ResourceStatusResponse:
    try:
        resource = container.resources.get(resource_id)
    except ResourceNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found",
        ) from exc

    current_version = container.versions.get_latest(resource_id)

    return ResourceStatusResponse(
        resource_id=resource.resource_id,
        source_url=resource.source_url,
        source_type=resource.source_type,
        status=resource.status,
        detected_language=resource.detected_language,
        language_confidence=resource.language_confidence,
        submitted_at=resource.submitted_at,
        updated_at=resource.updated_at,
       current_version=current_version.version_number if current_version is not None else None,
       error=resource.last_error,
   )

# TODO (junior dev): create the router and implement the endpoints.
#
#     router = APIRouter(prefix="/pipeline", tags=["pipeline"])
#
#
# POST /resources  ->  202 Accepted, SubmitResponse
#   [ ] Call SubmissionService.submit(). Nothing else.
#   [ ] RETURN IMMEDIATELY — never wait for the fetch. A synchronous fetch
#       here times out on large PDFs, the user retries, and now there are
#       duplicate submissions plus an annoyed user.
#   [ ] Map errors: not vetted -> 400 with a message pointing at
#       data/01_source_register; already submitted -> 200 with the existing id
#       (idempotent, not an error).
#
# POST /resources/upload  ->  202 Accepted
#   [ ] Accept an UploadFile for PDFs submitted by hand.
#   [ ] ENFORCE A MAX SIZE (settings.fetch_max_bytes) while streaming to disk.
#       FastAPI will happily buffer a 2GB upload into memory otherwise.
#   [ ] VERIFY THE MAGIC BYTES (%PDF-), not the filename. A filename is a
#       client-supplied string and means nothing.
#   [ ] Write to object storage and create the resource in FETCHED status,
#       enqueueing "extract" directly. There is nothing to fetch.
#
# GET /resources/{resource_id}  ->  ResourceStatusResponse
#   [ ] Status, timestamps, version count, safe error message.
#   [ ] 404 when unknown.
#
# GET /resources?status=...&limit=&offset=
#   [ ] Paginated. Cap `limit` server-side (say 200) — never let a client
#       request the entire table.
#
# GET /stats
#   [ ] Per-stage queue depth, counts per status, oldest item in review.
#       The 30-second "is it healthy?" view for the team's standup.
#
# DEPENDENCY INJECTION:
#   Use FastAPI `Depends` to get services from the Container built in the app
#   lifespan (see backend/main.py). Do NOT construct a Container per request —
#   that reloads the fastText and MT models on every call.
#
# AUTHENTICATION:
#   [ ] Put every route in this file behind auth before deployment. Start with
#       a shared API key header if that is all there is time for; an open
#       URL-fetching endpoint is a genuine security problem, not a to-do.
