"""
Tests for the stage template method, plus the ingest and extract stages.

**Owner: Dev A** (see the Sprint 1 split in `docs/PIPELINE_BACKLOG.md`).
Nobody else edits this file during Sprint 1 — that is what keeps four people
working in parallel without a daily merge conflict.

The template-method tests live here because Dev A owns `stages/base.py`. They
are written ONCE and cover behaviour every stage inherits, so the other two
stage test files do not repeat them.

WHAT TO TEST IN A STAGE: routing decisions and side effects.
  - Did it end in the right status?
  - Did it publish the right next job?
  - Did it call the ports it should have?

WHAT NOT TO TEST HERE: whether PyMuPDF parses a PDF correctly. That is the
adapter's business, and testing it here means these tests need real files and
get slow — at which point people stop running them.

End-to-end skeleton test for ingest and extract stages.

IMPLEMENTATION NOTES:
- This tests the complete flow: SUBMITTED -> FETCHED -> EXTRACTED
- Uses fake implementations for all dependencies
- Validates all intermediate states and data
"""

# These imports would be from the actual backend module
# For now, we'll use relative imports since this is a standalone implementation
import uuid
from pathlib import Path

from backend.modules.pipeline.adapters.queue.memory_queue import MemoryQueue
from backend.modules.pipeline.adapters.storage.filesystem_object_store import (
    FilesystemObjectStore,
)
from backend.modules.pipeline.domain.enums import ResourceStatus, SourceType
from backend.modules.pipeline.domain.models import Job, NormalizedDocument, TextBlock
from backend.modules.pipeline.registry import ExtractorRegistry, FetcherRegistry
from backend.modules.pipeline.stages.extract import ExtractStage
from backend.modules.pipeline.stages.ingest import IngestStage
from backend.tests.pipeline.fakes import (
    FakeDocumentRepository,
    FakeExtractor,
    FakeFetcher,
    FakeResourceRepository,
    FakeReviewRepository,
    MockDeduplicator,
    make_resource,
)


def build_test_container():
    """Build a test container with actual implementations where available."""
    resources = FakeResourceRepository()
    documents = FakeDocumentRepository()
    reviews = FakeReviewRepository()
    # Use actual MemoryQueue from repository
    queue = MemoryQueue()
    # Use actual FilesystemObjectStore from repository (temp directory for testing)
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    object_store = FilesystemObjectStore(root=temp_dir)
    dedup = MockDeduplicator()

    # Set up actual registries (now available in repository)
    fetcher_registry = FetcherRegistry()
    extractor_registry = ExtractorRegistry()

    # Register a fake web fetcher
    test_content = b"<html><body>Test content</body></html>"
    fetcher = FakeFetcher(
        source_type=SourceType.WEB,
        content=test_content,
        content_type="text/html",
        metadata={"title": "Test Page"},
    )
    fetcher_registry.register(fetcher)

    # Register a fake extractor
    test_document = NormalizedDocument(
        resource_id="",  # Will be set by extract stage
        title="Test Document",
        author="Test Author",
        published_date=None,
        blocks=(
            TextBlock(order=0, kind="heading", text="Test Heading"),
            TextBlock(order=1, kind="paragraph", text="Test paragraph content."),
        ),
        source_metadata={},
    )
    extractor = FakeExtractor(document=test_document)
    extractor_registry.register(extractor, priority=100)

    return {
        "resources": resources,
        "documents": documents,
        "reviews": reviews,
        "queue": queue,
        "object_store": object_store,
        "dedup": dedup,
        "fetcher_registry": fetcher_registry,
        "extractor_registry": extractor_registry,
    }


def test_ingest_extract_end_to_end():
    """Test complete flow: SUBMITTED -> FETCHED -> EXTRACTED."""
    container = build_test_container()

    # Create stages
    ingest_stage = IngestStage(
        resources=container["resources"],
        queue=container["queue"],
        reviews=container["reviews"],
        fetchers=container["fetcher_registry"],
        object_store=container["object_store"],
        deduplicator=container["dedup"],
    )

    extract_stage = ExtractStage(
        resources=container["resources"],
        queue=container["queue"],
        reviews=container["reviews"],
        documents=container["documents"],
        extractors=container["extractor_registry"],
        object_store=container["object_store"],
    )

    # Create a SUBMITTED resource
    resource_id = str(uuid.uuid4())
    resource = make_resource(
        resource_id=resource_id,
        source_type=SourceType.WEB,
        source_url="https://example.com/test",
        status=ResourceStatus.SUBMITTED,
    )
    container["resources"].add(resource)

    # Run ingest stage
    ingest_job = Job(
        job_id=str(uuid.uuid4()),
        resource_id=resource_id,
        stage="ingest",
    )
    ingest_stage.run(ingest_job)

    # Verify resource is now FETCHED
    updated_resource = container["resources"].get(resource_id)
    assert updated_resource.status == ResourceStatus.FETCHED
    assert updated_resource.raw_object_key is not None
    assert updated_resource.content_hash is not None

    # Verify object was stored
    stored_content = container["object_store"].get(updated_resource.raw_object_key)
    assert stored_content == b"<html><body>Test content</body></html>"

    # Verify extract job was enqueued
    # MemoryQueue uses consume() with context manager
    extract_job = None
    for job_handle in container["queue"].consume("extract", max_messages=1):
        with job_handle as job:
            extract_job = job
            assert job.resource_id == resource_id
    assert extract_job is not None, "Extract job should be enqueued"

    # Run extract stage
    extract_stage.run(extract_job)

    # Verify resource is now EXTRACTED
    final_resource = container["resources"].get(resource_id)
    assert final_resource.status == ResourceStatus.EXTRACTED

    # Verify document was saved
    document = container["documents"].get_document(resource_id)
    assert document is not None
    assert document.title == "Test Document"
    assert len(document.blocks) == 2

    # Verify detect_language job was enqueued
    detect_job = None
    for job_handle in container["queue"].consume("detect_language", max_messages=1):
        with job_handle as job:
            detect_job = job
            assert job.resource_id == resource_id
    assert detect_job is not None, "Detect language job should be enqueued"


def test_ingest_idempotency():
    """Test that ingest stage is idempotent - can handle re-delivered jobs."""
    container = build_test_container()

    ingest_stage = IngestStage(
        resources=container["resources"],
        queue=container["queue"],
        reviews=container["reviews"],
        fetchers=container["fetcher_registry"],
        object_store=container["object_store"],
        deduplicator=container["dedup"],
    )

    resource_id = str(uuid.uuid4())
    resource = make_resource(
        resource_id=resource_id,
        source_type=SourceType.WEB,
        source_url="https://example.com/test",
        status=ResourceStatus.SUBMITTED,
    )
    container["resources"].add(resource)

    # Run ingest once
    job1 = Job(job_id=str(uuid.uuid4()), resource_id=resource_id, stage="ingest")
    ingest_stage.run(job1)

    # Get the updated resource
    updated = container["resources"].get(resource_id)
    assert updated.status == ResourceStatus.FETCHED

    # Simulate a re-delivered job (resource already FETCHED)
    job2 = Job(job_id=str(uuid.uuid4()), resource_id=resource_id, stage="ingest")
    ingest_stage.run(job2)

    # Should be a no-op - status should still be FETCHED
    final = container["resources"].get(resource_id)
    assert final.status == ResourceStatus.FETCHED
    assert final.raw_object_key == updated.raw_object_key


def test_url_dedup():
    """Test that URL-level dedup prevents duplicate downloads."""
    container = build_test_container()

    ingest_stage = IngestStage(
        resources=container["resources"],
        queue=container["queue"],
        reviews=container["reviews"],
        fetchers=container["fetcher_registry"],
        object_store=container["object_store"],
        deduplicator=container["dedup"],
    )

    url = "https://example.com/duplicate-test"

    # First resource
    resource1 = make_resource(
        resource_id=str(uuid.uuid4()),
        source_type=SourceType.WEB,
        source_url=url,
        status=ResourceStatus.SUBMITTED,
    )
    container["resources"].add(resource1)

    job1 = Job(job_id=str(uuid.uuid4()), resource_id=resource1.resource_id, stage="ingest")
    ingest_stage.run(job1)

    assert container["resources"].get(resource1.resource_id).status == ResourceStatus.FETCHED

    # Second resource with same URL
    resource2 = make_resource(
        resource_id=str(uuid.uuid4()),
        source_type=SourceType.WEB,
        source_url=url,
        status=ResourceStatus.SUBMITTED,
    )
    container["resources"].add(resource2)

    job2 = Job(job_id=str(uuid.uuid4()), resource_id=resource2.resource_id, stage="ingest")
    ingest_stage.run(job2)

    # Should be marked as DUPLICATE
    assert container["resources"].get(resource2.resource_id).status == ResourceStatus.DUPLICATE


if __name__ == "__main__":
    # Run tests manually for verification
    print("Running test_ingest_extract_end_to_end...")
    test_ingest_extract_end_to_end()
    print("✓ test_ingest_extract_end_to_end passed")

    print("Running test_ingest_idempotency...")
    test_ingest_idempotency()
    print("✓ test_ingest_idempotency passed")

    print("Running test_url_dedup...")
    test_url_dedup()
    print("✓ test_url_dedup passed")

    print("\nAll tests passed!")
