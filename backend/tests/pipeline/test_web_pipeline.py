from __future__ import annotations

import socket

import httpx
import pytest
from backend.modules.pipeline.adapters.extractors.html_extractor import HtmlExtractor
from backend.modules.pipeline.api.routes_pipeline import router
from backend.modules.pipeline.container import build_test_container
from backend.modules.pipeline.domain.models import NormalizedDocument, TextBlock
from backend.modules.pipeline.registry import ExtractorRegistry, FetcherRegistry
from backend.modules.pipeline.stages.extract import ExtractStage
from backend.modules.pipeline.stages.ingest import IngestStage
from backend.tests.pipeline.fakes import (
    FakeDeduplicator,
    FakeDocumentRepository,
    FakeExtractor,
    FakeFetcher,
    FakeJobQueue,
    FakeObjectStore,
    FakeResourceRepository,
    FakeReviewRepository,
    make_resource,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules.pipeline.adapters.fetchers.web_fetcher import WebFetcher
from modules.pipeline.domain.enums import ResourceStatus, SourceType
from modules.pipeline.domain.errors import PermanentError
from modules.pipeline.domain.models import Job


def make_fetcher(
    handler,
    max_bytes: int = 10_000,
) -> WebFetcher:
    return WebFetcher(
        timeout_seconds=5.0,
        max_bytes=max_bytes,
        user_agent="MamaCareAI-Test/1.0",
        respect_robots=False,
        transport=httpx.MockTransport(handler),
    )


def make_ingest_stage(
    *,
    content: bytes = b"<html><body>Test content</body></html>",
    deduplicator=None,
) -> tuple[IngestStage, FakeResourceRepository, FakeJobQueue, FakeDeduplicator]:
    resources = FakeResourceRepository()
    queue = FakeJobQueue()
    reviews = FakeReviewRepository()
    object_store = FakeObjectStore()

    dedup = deduplicator or FakeDeduplicator()

    fetchers = FetcherRegistry()
    fetchers.register(
        FakeFetcher(
            source_type=SourceType.WEB,
            content=content,
            content_type="text/html",
        )
    )

    stage = IngestStage(
        resources=resources,
        queue=queue,
        reviews=reviews,
        fetchers=fetchers,
        object_store=object_store,
        deduplicator=dedup,
    )

    return stage, resources, queue, dedup



def public_dns(hostname, *args, **kwargs):
    return [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            ("93.184.216.34", 0),
        )
    ]


def test_safe_redirect_is_followed(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(
                302,
                headers={"Location": "https://example.com/final"},
                request=request,
            )

        return httpx.Response(
            200,
            content=b"<html><body>Safe page</body></html>",
            request=request,
        )

    result = make_fetcher(handler).fetch("https://example.com/start")

    assert result.content == b"<html><body>Safe page</body></html>"
    assert result.metadata["final_url"] == "https://example.com/final"


def test_redirect_to_private_ip_is_rejected(monkeypatch) -> None:
    def fake_getaddrinfo(hostname, *args, **kwargs):
        if hostname == "private.local":
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("127.0.0.1", 0),
                )
            ]

        return public_dns(hostname, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "http://private.local/internal"},
            request=request,
        )

    with pytest.raises(PermanentError, match="private or local"):
        make_fetcher(handler).fetch("https://example.com/start")


def test_too_many_redirects_are_rejected(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": str(request.url)},
            request=request,
        )

    with pytest.raises(PermanentError, match="Too many redirects"):
        make_fetcher(handler).fetch("https://example.com/start")


def test_initial_private_ip_is_rejected(monkeypatch) -> None:
    def fake_getaddrinfo(hostname, *args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("127.0.0.1", 0),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Private destination must never be requested")

    with pytest.raises(PermanentError, match="private or local"):
        make_fetcher(handler).fetch("http://internal.test/data")


def test_unsupported_scheme_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Unsupported schemes must never be requested")

    with pytest.raises(PermanentError, match="Unsupported URL scheme"):
        make_fetcher(handler).fetch("ftp://example.com/file")


def test_size_limit_is_enforced(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"x" * 101,
            request=request,
        )

    with pytest.raises(PermanentError, match="max_bytes"):
      make_fetcher(
        max_bytes=100,
        handler=handler,
    ).fetch("https://example.com/large")
def test_url_duplicate_is_rejected_before_download() -> None:
    stage, resources, _, dedup = make_ingest_stage()

    url = "https://example.com/same"

    # Mark the URL hash as already known.
    url_hash = dedup.compute_hash(
        source_url=url,
        content=b"",
    )
    dedup.is_duplicate(url_hash)

    resource = make_resource(
        source_type=SourceType.WEB,
        source_url=url,
        status=ResourceStatus.SUBMITTED,
    )
    resources.save(resource)

    job = Job(
        job_id="job-url-duplicate",
        resource_id=resource.resource_id,
        stage="ingest",
    )

    stage.run(job)

    updated = resources.get(resource.resource_id)

    assert updated.status == ResourceStatus.DUPLICATE
    assert updated.raw_object_key is None

def test_content_duplicate_is_rejected_after_download() -> None:
    content = b"<html><body>Same document</body></html>"
    stage, resources, _, dedup = make_ingest_stage(
        content=content,
    )

    url = "https://example.com/new-url"
    # Pretend this exact document was already seen.
    content_hash = dedup.compute_hash(
        source_url=url,
        content=content,
    )
    dedup.is_duplicate(content_hash)

    resource = make_resource(
        source_type=SourceType.WEB,
        source_url=url,
        status=ResourceStatus.SUBMITTED,
    )
    resources.save(resource)

    job = Job(
        job_id="job-content-duplicate",
        resource_id=resource.resource_id,
        stage="ingest",
    )
    stage.run(job)
    updated = resources.get(resource.resource_id)

    assert updated.status == ResourceStatus.DUPLICATE
    assert updated.raw_object_key is None
def make_extract_stage(
    document: NormalizedDocument,
) -> tuple[ExtractStage, FakeResourceRepository, FakeDocumentRepository, FakeJobQueue]:
    resources = FakeResourceRepository()
    documents = FakeDocumentRepository()
    reviews = FakeReviewRepository()
    queue = FakeJobQueue()
    object_store = FakeObjectStore()

    object_key = "raw/test-resource/web"
    object_store.put(
        object_key,
        b"<html><body>test content</body></html>",
        content_type="text/html",
    )

    extractors = ExtractorRegistry()
    extractors.register(
        FakeExtractor(document=document),
        priority=100,
    )

    stage = ExtractStage(
        resources=resources,
        queue=queue,
        reviews=reviews,
        documents=documents,
        extractors=extractors,
        object_store=object_store,
    )
    return stage, resources, documents, queue
def test_extraction_rejects_unusable_short_document() -> None:
    document = NormalizedDocument(
        resource_id="",
        title="Too Short",
        author=None,
        published_date=None,
        blocks=(
            TextBlock(
                order=0,
                kind="paragraph",
                text="Too short",
            ),
        ),
        source_metadata={},
    )

    stage, resources, _, queue = make_extract_stage(document)

    resource = make_resource(
        resource_id="short-doc",
        source_type=SourceType.WEB,
        status=ResourceStatus.FETCHED,
        raw_object_key="raw/test-resource/web",
    )
    resources.save(resource)

    job = Job(
        job_id="job-short-doc",
        resource_id=resource.resource_id,
        stage="extract",
    )

    stage.run(job)

    updated = resources.get(resource.resource_id)

    assert updated.status == ResourceStatus.FAILED
    assert queue.depth("detect_language") == 0

def test_extraction_rejects_noncontiguous_block_order() -> None:
    document = NormalizedDocument(
        resource_id="",
        title="Bad Order",
        author=None,
        published_date=None,
        blocks=(
            TextBlock(order=0, kind="heading", text="Heading"),
            TextBlock(
                order=2,
                kind="paragraph",
                text="This document has a missing block order.",
            ),
        ),
        source_metadata={},
    )

    stage, resources, _, queue = make_extract_stage(document)

    resource = make_resource(
        resource_id="bad-order",
        source_type=SourceType.WEB,
        status=ResourceStatus.FETCHED,
        raw_object_key="raw/test-resource/web",
    )
    resources.save(resource)

    job = Job(
        job_id="job-bad-order",
        resource_id=resource.resource_id,
        stage="extract",
    )

    stage.run(job)

    updated = resources.get(resource.resource_id)

    assert updated.status == ResourceStatus.FAILED
    assert queue.depth("detect_language") == 0

def test_extraction_preserves_original_text_blocks() -> None:
    original_blocks = (
        TextBlock(
            order=0,
            kind="heading",
            text="Maternal Health Guide",
        ),
        TextBlock(
            order=1,
            kind="paragraph",
            text="This is the original paragraph and it must remain unchanged.",
        ),
    )

    document = NormalizedDocument(
        resource_id="",
        title="Maternal Health Guide",
        author="Test Author",
        published_date=None,
        blocks=original_blocks,
        source_metadata={"language": "en"},
    )

    stage, resources, documents, _ = make_extract_stage(document)

    resource = make_resource(
        resource_id="preserve-text",
        source_type=SourceType.WEB,
        status=ResourceStatus.FETCHED,
        raw_object_key="raw/test-resource/web",
    )
    resources.save(resource)

    job = Job(
        job_id="job-preserve-text",
        resource_id=resource.resource_id,
        stage="extract",
    )

    stage.run(job)

    saved = documents.get_document(resource.resource_id)

    assert saved.title == document.title
    assert saved.author == document.author
    assert saved.blocks == original_blocks
    assert saved.source_metadata == document.source_metadata
def test_html_extraction_filters_boilerplate_and_preserves_article_order() -> None:
    html = b"""
    <!doctype html>
    <html>
    <body>
      <header>
        <div>Maternal Health Portal</div>
      </header>

      <nav>
        <a href="/">Home</a>
        <a href="/guides">Guides</a>
        <a href="/contact">Contact</a>
      </nav>

      <div id="cookie-banner">
        <p>We use cookies on this site.</p>
        <button>Accept cookies</button>
      </div>

      <main>
        <h1>Pregnancy Danger Signs</h1>
        <p>
          Severe bleeding during pregnancy requires immediate assessment at
          a health facility. Do not wait for the symptoms to become worse.
        </p>
        <h2>When to seek urgent care</h2>
        <p>
          Seek urgent care when there is severe bleeding, convulsions, or
          severe headache that does not improve with rest.
        </p>
      </main>

      <footer>
        Copyright 2026 Maternal Health Portal
      </footer>
    </body>
    </html>
    """

    document = HtmlExtractor().extract(
        "html-boilerplate",
        html,
        metadata={"content_type": "text/html; charset=utf-8"},
    )

    assert [block.order for block in document.blocks] == list(
        range(len(document.blocks))
    )

    texts = [block.text.lower() for block in document.blocks]
    body_text = " ".join(texts)

    assert "pregnancy danger signs" in body_text
    assert "severe bleeding during pregnancy" in body_text
    assert "seek urgent care when there is severe bleeding" in body_text

    assert "maternal health portal" not in body_text
    assert "home" not in body_text
    assert "guides" not in body_text
    assert "contact" not in body_text
    assert "we use cookies on this site" not in body_text
    assert "accept cookies" not in body_text
    assert "copyright 2026" not in body_text

    assert document.blocks[0].kind == "heading"
    assert document.blocks[0].text == "Pregnancy Danger Signs"


def test_html_extraction_removes_repeated_boilerplate_but_keeps_useful_text() -> None:
    html = b"""
    <!doctype html>
    <html>
    <body>
      <main>
        <p>
          This article explains warning signs during pregnancy and when a
          pregnant woman should seek immediate medical attention.
        </p>

        <div class="repeated-footer">
          Privacy Policy
        </div>

        <p>
          The guidance recommends contacting a health facility promptly if
          symptoms become severe or unexpected.
        </p>

        <div class="repeated-footer">
          Privacy Policy
        </div>

        <p>
          Patients should keep emergency contact information available and
          follow instructions given by qualified health professionals.
        </p>
      </main>
    </body>
    </html>
    """

    document = HtmlExtractor().extract(
        "html-repeated-boilerplate",
        html,
        metadata={"content_type": "text/html; charset=utf-8"},
    )

    body_text = " ".join(block.text for block in document.blocks)

    assert "This article explains warning signs during pregnancy" in body_text
    assert "The guidance recommends contacting a health facility promptly" in body_text
    assert "Patients should keep emergency contact information available" in body_text

    # The extractor should not duplicate repeated page furniture into the
    # normalized document.
    assert body_text.lower().count("privacy policy") <= 1


def test_pipeline_routes_require_api_key(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_API_KEY", "test-secret")

    app = FastAPI()
    app.state.container = build_test_container()
    app.include_router(router)

    client = TestClient(app)

    response = client.get(
        "/pipeline/stats",
    )
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "pipeline_auth_invalid"


def test_pipeline_routes_accept_valid_api_key(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_API_KEY", "test-secret")

    app = FastAPI()
    app.state.container = build_test_container()
    app.include_router(router)

    client = TestClient(app)

    response = client.get(
        "/pipeline/stats",
        headers={"X-API-Key": "test-secret"},
    )

    assert response.status_code == 200


def test_pipeline_routes_fail_closed_when_auth_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_API_KEY", raising=False)

    app = FastAPI()
    app.state.container = build_test_container()
    app.include_router(router)

    client = TestClient(app)

    response = client.get(
        "/pipeline/stats",
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "pipeline_auth_not_configured"
