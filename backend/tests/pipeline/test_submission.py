import pytest

from modules.pipeline.domain.enums import ResourceStatus, SourceType
from modules.pipeline.domain.errors import PermanentError
from modules.pipeline.services.submission import SubmissionService

from .fakes import FakeJobQueue, FakeResourceRepository


class FakeSourceRegister:
    def __init__(self, approved: bool = True):
        self.approved = approved

    def is_approved(self, source_url: str) -> bool:
        return self.approved


def create_service(approved: bool = True):
    resources = FakeResourceRepository()
    queue = FakeJobQueue()
    source_register = FakeSourceRegister(approved=approved)

    service = SubmissionService(
        resources=resources,
        queue=queue,
        source_register=source_register,
    )

    return service, resources, queue


def test_submit_rejects_non_http_url():
    service, _, _ = create_service()

    with pytest.raises(PermanentError):
        service.submit(
            source_url="ftp://example.com/file.pdf",
            submitted_by="test-user",
        )


def test_submit_rejects_unvetted_source():
    service, _, _ = create_service(approved=False)

    with pytest.raises(PermanentError, match="not vetted"):
        service.submit(
            source_url="https://example.com/file.pdf",
            submitted_by="test-user",
        )


def test_submit_creates_pdf_resource_and_ingest_job():
    service, resources, queue = create_service()

    resource = service.submit(
        source_url="https://example.com/guide.pdf",
        submitted_by="test-user",
        metadata={"title": "Maternal Guide"},
    )

    assert resource.source_type == SourceType.PDF
    assert resource.source_url == "https://example.com/guide.pdf"
    assert resource.status == ResourceStatus.SUBMITTED
    assert resources.get(resource.resource_id) == resource
    assert queue.depth("ingest") == 1


def test_submit_infers_pdf_source_type():
    service, _, _ = create_service()

    resource = service.submit(
        source_url="https://example.com/maternal-guide.pdf",
        submitted_by="test-user",
    )

    assert resource.source_type == SourceType.PDF


def test_explicit_source_type_wins_over_inference():
    service, _, _ = create_service()

    resource = service.submit(
        source_url="https://example.com/maternal-guide.pdf",
        source_type=SourceType.WEB,
        submitted_by="test-user",
    )

    assert resource.source_type == SourceType.WEB


def test_submit_rejects_loopback_url():
    service, _, _ = create_service()

    with pytest.raises(PermanentError):
        service.submit(
            source_url="http://127.0.0.1",
            submitted_by="test-user",
        )


def test_submit_preserves_metadata():
    service, _, _ = create_service()

    resource = service.submit(
        source_url="https://example.com/guide.pdf",
        submitted_by="david",
        metadata={"publisher": "WHO"},
    )

    assert resource.source_metadata["publisher"] == "WHO"
    assert resource.source_metadata["submitted_by"] == "david"   