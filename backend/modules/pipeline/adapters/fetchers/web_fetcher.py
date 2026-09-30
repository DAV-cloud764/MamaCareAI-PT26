"""
Web fetcher (PDF 3.1) — "a crawler/fetcher service (Scrapy, or Playwright for
JavaScript-heavy pages) pulls HTML, respects robots.txt, and pushes raw HTML to
object storage".

TWO STRATEGIES, AND WHEN EACH IS RIGHT
  - Plain HTTP (httpx): fast, cheap, works for most static health-ministry and
    NGO pages. START HERE.
  - Playwright: renders JavaScript. Necessary for SPA-style sites, but it costs
    a browser process per page — roughly 100x the resources. Use it only when
    the plain fetch demonstrably returns an empty shell.

Do not reach for Playwright by default. Measure first: fetch the page plainly,
and if the extracted text is near-empty, fall back. That fallback is a
priority-ordered registry decision, not an if-statement in this file.

ROBOTS.TXT IS NOT OPTIONAL. `settings.respect_robots_txt` defaults to True and
must stay True. We are building a public-health resource for a named
organization; ignoring robots.txt is both an ethical and a reputational
problem, and it is exactly the kind of thing that gets the project's access
revoked at the worst moment.
"""

from __future__ import annotations

import ipaddress
import socket
import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse

import httpx

from ...domain.enums import SourceType
from ...domain.errors import FetchError, PermanentError, ProviderRateLimited
from ...ports.fetcher import FetchResult, SourceFetcher


class WebFetcher(SourceFetcher):
    """Fetches HTML over plain HTTP."""

    _MAX_REDIRECTS = 5

    def __init__(
        self,
        *,
        timeout_seconds: float,
        max_bytes: int,
        user_agent: str,
        respect_robots: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._max_bytes = max_bytes
        self._user_agent = user_agent
        self._respect_robots = respect_robots
        self._client = httpx.Client(
            timeout=timeout_seconds,
            headers={"User-Agent": user_agent},
            follow_redirects=False,
            transport=transport,
        )
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_request_time: dict[str, float] = {}
        self._min_delay_seconds = 1.0
    def fetch(self, source_url: str) -> FetchResult:
        # Validate the original destination before making any request.
        self._validate_public_url(source_url)

        original_domain = urlparse(source_url).netloc

        if self._respect_robots and original_domain:
            parser = self._get_robots_parser(original_domain)
            if not parser.can_fetch(self._user_agent, source_url):
                raise PermanentError(
                    f"robots.txt disallows fetching {source_url}"
                )

        current_url = source_url
        redirect_count = 0

        while True:
            domain = urlparse(current_url).netloc
            self._respect_rate_limit(domain)

            try:
                with self._client.stream("GET", current_url) as response:

                    # Handle redirects before raise_for_status().
                    if response.status_code in {
                        301,
                        302,
                        303,
                        307,
                        308,
                    }:
                        if redirect_count >= self._MAX_REDIRECTS:
                            raise PermanentError(
                                f"Too many redirects fetching {source_url}"
                            )

                        location = response.headers.get("Location")

                        if not location:
                            raise PermanentError(
                                f"Redirect response missing Location: {current_url}"
                            )

                        next_url = urljoin(current_url, location)

                        # Validate the redirect destination before following it.
                        self._validate_public_url(next_url)

                        redirect_count += 1
                        current_url = next_url
                        continue

                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        retry_after_seconds = (
                            float(retry_after) if retry_after else None
                        )
                        raise ProviderRateLimited(
                            f"Rate limited fetching {current_url}",
                            retry_after_seconds=retry_after_seconds,
                        )

                    if response.status_code in (404, 403):
                        raise PermanentError(
                            f"{response.status_code} fetching {current_url}"
                        )

                    if response.status_code >= 500:
                        raise FetchError(
                            f"{response.status_code} fetching {current_url}"
                        )

                    response.raise_for_status()

                    chunks: list[bytes] = []
                    total = 0

                    for chunk in response.iter_bytes():
                        total += len(chunk)

                        if total > self._max_bytes:
                            raise PermanentError(
                                f"{source_url} exceeded max_bytes={self._max_bytes}"
                            )

                        chunks.append(chunk)

                    content = b"".join(chunks)

                    metadata: dict[str, object] = {
                        "final_url": str(response.url),
                        "content_type": response.headers.get(
                            "content-type",
                            "",
                        ),
                        "last_modified": response.headers.get(
                            "last-modified"
                        ),
                        "etag": response.headers.get("etag"),
                    }

                    return FetchResult(
                        content=content,
                        content_type=response.headers.get(
                            "content-type",
                            "text/html",
                        ),
                        metadata=metadata,
                    )

            except httpx.TimeoutException as exc:
                raise FetchError(
                    f"Timeout fetching {current_url}"
                ) from exc

            except httpx.TransportError as exc:
                raise FetchError(
                    f"Connection error fetching {current_url}"
                ) from exc
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_request_time: dict[str, float] = {}
        self._min_delay_seconds = 1.0

    @property
    def source_type(self) -> SourceType:
        return SourceType.WEB

    @staticmethod
    def _validate_public_url(source_url: str) -> None:
        """Reject unsupported schemes and hosts resolving to private addresses."""
        parsed = urlparse(source_url)

        if parsed.scheme.lower() not in {"http", "https"}:
            raise PermanentError(
                f"Unsupported URL scheme for {source_url}"
            )

        hostname = parsed.hostname

        if not hostname:
            raise PermanentError(
                f"URL has no hostname: {source_url}"
            )

        try:
            addresses = socket.getaddrinfo(
                hostname,
                None,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise PermanentError(
                f"URL hostname could not be resolved: {hostname}"
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
                    f"URL resolves to a private or local address: {hostname}")
    def _get_robots_parser(self, domain: str) -> urllib.robotparser.RobotFileParser:
        if domain not in self._robots_cache:
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(f"https://{domain}/robots.txt")
            try:
                parser.read()
            except Exception:  # noqa: BLE001, S110 - intentional: fail open
                # An unreachable robots.txt must not block every fetch on a
                # domain that has no robots.txt at all.
                pass
            self._robots_cache[domain] = parser
        return self._robots_cache[domain]

    def _respect_rate_limit(self, domain: str) -> None:
        last = self._last_request_time.get(domain)
        if last is not None:
            elapsed = time.monotonic() - last
            if elapsed < self._min_delay_seconds:
                time.sleep(self._min_delay_seconds - elapsed)
        self._last_request_time[domain] = time.monotonic()
