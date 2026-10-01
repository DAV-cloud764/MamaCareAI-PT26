"""
WebFetcher tests (PIPE-17) — httpx.MockTransport, no real network calls.

What is covered: status-code mapping (2xx / 404 / 403 / 5xx / 429), the
max_bytes size cap, and the timeout path. The robots.txt and rate-limit rules
are tested through `respect_robots=False` here because fetching robots.txt goes
over urllib (not httpx), so it cannot be mocked by MockTransport — robots logic
is real-network and reviewed manually / via the integration PR.
"""

from __future__ import annotations

import httpx
import pytest

from modules.pipeline.adapters.fetchers.web_fetcher import WebFetcher
from modules.pipeline.domain.errors import (
    FetchError,
    PermanentError,
    ProviderRateLimited,
)


def make_fetcher(*, max_bytes: int = 10_000, handler) -> WebFetcher:
    return WebFetcher(
        timeout_seconds=5.0,
        max_bytes=max_bytes,
        user_agent="MamaCareAI-Test/1.0",
        respect_robots=False,
        transport=httpx.MockTransport(handler),
    )


def test_200_returns_content_and_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://health.gov/article"
        assert request.headers["User-Agent"] == "MamaCareAI-Test/1.0"
        return httpx.Response(
            200,
            content=b"<html><body>Hello</body></html>",
            headers={
                "content-type": "text/html; charset=utf-8",
                "etag": '"abc123"',
                "last-modified": "Wed, 01 Jan 2026 00:00:00 GMT",
            },
            request=request,
        )

    result = make_fetcher(handler=handler).fetch("https://health.gov/article")

    assert result.content == b"<html><body>Hello</body></html>"
    assert result.content_type == "text/html; charset=utf-8"
    assert result.metadata["etag"] == '"abc123"'
    assert result.metadata["last_modified"] is not None
    assert result.metadata["final_url"] == "https://health.gov/article"


def test_404_raises_permanent_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"not found", request=request)

    with pytest.raises(PermanentError):
        make_fetcher(handler=handler).fetch("https://health.gov/missing")


def test_403_raises_permanent_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, content=b"forbidden", request=request)

    with pytest.raises(PermanentError):
        make_fetcher(handler=handler).fetch("https://health.gov/private")


def test_500_raises_retryable_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom", request=request)

    with pytest.raises(FetchError) as exc_info:
        make_fetcher(handler=handler).fetch("https://health.gov/flaky")

    assert exc_info.value.retryable


def test_429_raises_provider_rate_limited_with_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "42"},
            request=request,
        )

    with pytest.raises(ProviderRateLimited) as exc_info:
        make_fetcher(handler=handler).fetch("https://health.gov/busy")

    assert exc_info.value.retry_after_seconds == 42.0


def test_content_over_max_bytes_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"x" * 20_000,
            request=request,
        )

    with pytest.raises(PermanentError) as exc_info:
        make_fetcher(max_bytes=1_000, handler=handler).fetch("https://health.gov/big")

    assert "max_bytes" in str(exc_info.value)


def test_network_timeout_raises_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connection timed out", request=request)

    with pytest.raises(FetchError) as exc_info:
        make_fetcher(handler=handler).fetch("https://health.gov/slow")

    assert exc_info.value.retryable
