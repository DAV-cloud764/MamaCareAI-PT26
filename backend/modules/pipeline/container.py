from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from .config import PipelineSettings
from .domain.errors import InvalidStateTransition
from .ports.deduplicator import Deduplicator
from .ports.job_queue import JobQueue
from .ports.language_detector import LanguageDetector
from .ports.object_store import ObjectStore
from .ports.repositories import (
    DocumentRepository,
    ResourceRepository,
    ReviewRepository,
    VersionRepository,
)
from .ports.search_index import SearchIndex
from .ports.translator import Translator
from .registry import ExtractorRegistry, FetcherRegistry
from .stages.base import Stage


@dataclass(slots=True)
class Container:
    settings: PipelineSettings
    queue: JobQueue
    object_store: ObjectStore
    search: SearchIndex
    deduplicator: Deduplicator
    detector: LanguageDetector
    translator: Translator
    fetchers: FetcherRegistry
    extractors: ExtractorRegistry
    resources: ResourceRepository
    documents: DocumentRepository
    versions: VersionRepository
    reviews: ReviewRepository

    def stage(self, name: str) -> Stage:
        from .adapters.translation.chunker import Chunker
        from .observability.metrics import NullMetrics, PrometheusMetrics
        from .services.compliance import ComplianceGate
        from .services.review_service import ReviewService
        from .stages.detect_language import DetectLanguageStage
        from .stages.extract import ExtractStage
        from .stages.ingest import IngestStage
        from .stages.publish import PublishStage
        from .stages.review import ReviewStage
        from .stages.store import StoreStage
        from .stages.translate import TranslateStage

        # PrometheusMetrics is a safe no-op without prometheus-client
        # installed (see observability/metrics.py), so this never breaks
        # tests or clean checkouts. Stages default to NullMetrics anyway.
        metrics = (
            PrometheusMetrics()
            if self.settings.metrics_enabled
            else NullMetrics()
        )
        common = {
            "resources": self.resources,
            "queue": self.queue,
            "reviews": self.reviews,
            "max_attempts": self.settings.max_attempts,
            "metrics": metrics,
        }
        review_service = ReviewService(
            resources=self.resources,
            reviews=self.reviews,
            versions=self.versions,
            documents=self.documents,
            queue=self.queue,
        )
        stages = {
            "ingest": lambda: IngestStage(**common, fetchers=self.fetchers, object_store=self.object_store, deduplicator=self.deduplicator),
            "extract": lambda: ExtractStage(**common, documents=self.documents, extractors=self.extractors, object_store=self.object_store),
            "detect_language": lambda: DetectLanguageStage(**common, documents=self.documents, detector=self.detector, confidence_threshold=self.settings.language_confidence_threshold, target_language=self.settings.target_language),
            "translate": lambda: TranslateStage(**common, documents=self.documents, versions=self.versions, translator=self.translator, chunker=Chunker(max_chars=self.settings.translation_max_chunk_chars), target_language=self.settings.target_language),
            "store": lambda: StoreStage(**common, documents=self.documents, versions=self.versions, search=self.search, review_service=review_service),
            "review": lambda: ReviewStage(**common),
            "publish": lambda: PublishStage(
                **common,
                versions=self.versions,
                search=self.search,
                compliance_gate=ComplianceGate(
                    strict=self.settings.compliance_strict,
                    allowed_licenses=self.settings.allowed_license_set(),
                ),
            ),
        }
        try:
            return stages[name]()
        except KeyError as exc:
            valid = ", ".join(stages)
            raise ValueError(f"Unknown pipeline stage {name!r}. Valid stages: {valid}") from exc


def build_queue(settings: PipelineSettings) -> JobQueue:
    if settings.queue_backend == "memory":
        from .adapters.queue.memory_queue import MemoryQueue
        return MemoryQueue()
    if settings.queue_backend == "sqs":
        from .adapters.queue.sqs_queue import SqsQueue
        queue_urls = {
            stage: url for stage in ("ingest", "extract", "detect_language", "translate", "store", "review", "publish")
            if (url := os.getenv(f"PIPELINE_SQS_{stage.upper()}_QUEUE_URL"))
        }
        dead_letter_url = os.getenv("PIPELINE_SQS_DEAD_LETTER_URL")
        region = os.getenv("PIPELINE_AWS_REGION")
        if len(queue_urls) != 7 or not dead_letter_url or not region:
            raise ValueError("SQS requires PIPELINE_AWS_REGION, PIPELINE_SQS_DEAD_LETTER_URL, and one PIPELINE_SQS_<STAGE>_QUEUE_URL per stage")
        return SqsQueue(queue_urls=queue_urls, dead_letter_url=dead_letter_url, region=region)
    raise ValueError(f"Unsupported queue backend: {settings.queue_backend!r}. Supported: memory, sqs")


def build_object_store(settings: PipelineSettings) -> ObjectStore:
    if settings.object_store_backend == "filesystem":
        from .adapters.storage.filesystem_object_store import FilesystemObjectStore
        return FilesystemObjectStore(root=Path(settings.object_store_path))
    if settings.object_store_backend == "s3":
        from .adapters.storage.s3_object_store import S3ObjectStore
        bucket = os.getenv("PIPELINE_S3_BUCKET")
        if not bucket:
            raise ValueError("S3 object storage requires PIPELINE_S3_BUCKET")
        return S3ObjectStore(bucket=bucket, region=os.getenv("PIPELINE_AWS_REGION"), endpoint_url=os.getenv("PIPELINE_S3_ENDPOINT_URL"), prefix=os.getenv("PIPELINE_S3_PREFIX", ""))
    raise ValueError(f"Unsupported object store backend: {settings.object_store_backend!r}. Supported: filesystem, s3")


def build_search_index(settings: PipelineSettings) -> SearchIndex:
    if settings.search_backend == "sqlite":
        from .adapters.storage.sqlite_search_index import SqliteSearchIndex
        database_path = settings.database_url.removeprefix("sqlite:///")
        return SqliteSearchIndex(database_path=database_path)
    if settings.search_backend == "opensearch":
        from .adapters.storage.opensearch_index import OpenSearchIndex
        hosts = [host.strip() for host in os.getenv("PIPELINE_OPENSEARCH_HOSTS", "").split(",") if host.strip()]
        if not hosts:
            raise ValueError("OpenSearch requires PIPELINE_OPENSEARCH_HOSTS")
        return OpenSearchIndex(hosts=hosts, index_name=os.getenv("PIPELINE_OPENSEARCH_INDEX", "mamacare-resources"), username=os.getenv("PIPELINE_OPENSEARCH_USERNAME"), password=os.getenv("PIPELINE_OPENSEARCH_PASSWORD"))
    raise ValueError(f"Unsupported search backend: {settings.search_backend!r}. Supported: sqlite, opensearch")


def build_translator(settings: PipelineSettings) -> Translator:
    if settings.translation_engine == "passthrough":
        from .adapters.translation.passthrough_translator import PassthroughTranslator
        return PassthroughTranslator()
    if settings.translation_engine == "nllb":
        from .adapters.translation.nllb_translator import NllbTranslator
        return NllbTranslator(batch_size=settings.translation_batch_size)
    if settings.translation_engine == "gemini":
        from .adapters.translation.gemini_translator import GeminiTranslator
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or settings.translation_api_key
            or os.getenv("PIPELINE_TRANSLATION_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "gemini translation requires PIPELINE_TRANSLATION_API_KEY "
                "or GEMINI_API_KEY"
            )
        return GeminiTranslator(
            api_key=api_key,
            model=settings.gemini_model,
            batch_size=settings.translation_batch_size,
            timeout_seconds=settings.gemini_timeout_seconds,
        )
    if settings.translation_engine in {"google", "aws", "azure"}:
        from .adapters.translation.cloud_translator import CloudTranslator
        api_key = os.getenv("PIPELINE_TRANSLATION_API_KEY")
        if not api_key:
            raise ValueError(f"{settings.translation_engine} translation requires PIPELINE_TRANSLATION_API_KEY")
        return CloudTranslator(provider=settings.translation_engine, api_key=api_key, region=os.getenv("PIPELINE_AWS_REGION"), batch_size=settings.translation_batch_size)
    raise ValueError(f"Unsupported translation engine: {settings.translation_engine!r}. Supported: passthrough, nllb, gemini, google, aws, azure")


def build_detector(settings: PipelineSettings) -> LanguageDetector:
    if settings.language_detector == "fasttext":
        from .adapters.language.fasttext_detector import FastTextDetector
        return FastTextDetector(model_path=os.getenv("PIPELINE_FASTTEXT_MODEL_PATH", "./models/lid.176.bin"))
    raise ValueError(f"Unsupported language detector: {settings.language_detector!r}. Supported: fasttext")


def build_fetchers(settings: PipelineSettings) -> FetcherRegistry:
    from .adapters.fetchers.pdf_fetcher import PdfFetcher
    from .adapters.fetchers.video_fetcher import VideoFetcher
    from .adapters.fetchers.web_fetcher import WebFetcher

    registry = FetcherRegistry()
    registry.register(WebFetcher(timeout_seconds=settings.fetch_timeout_seconds, max_bytes=settings.fetch_max_bytes, user_agent=settings.user_agent, respect_robots=settings.respect_robots_txt))
    registry.register(PdfFetcher(timeout_seconds=settings.fetch_timeout_seconds, max_bytes=settings.fetch_max_bytes, user_agent=settings.user_agent))
    registry.register(VideoFetcher(timeout_seconds=settings.fetch_timeout_seconds, max_bytes=settings.fetch_max_bytes))
    return registry


def build_extractors(settings: PipelineSettings) -> ExtractorRegistry:
    from .adapters.extractors.asr_extractor import AsrExtractor
    from .adapters.extractors.caption_extractor import CaptionExtractor
    from .adapters.extractors.html_extractor import HtmlExtractor
    from .adapters.extractors.pdf_ocr_extractor import PdfOcrExtractor
    from .adapters.extractors.pdf_text_extractor import PdfTextExtractor

    registry = ExtractorRegistry()
    registry.register(HtmlExtractor(), priority=100)
    registry.register(PdfTextExtractor(), priority=100)
    registry.register(PdfOcrExtractor(), priority=50)
    registry.register(CaptionExtractor(), priority=100)
    registry.register(AsrExtractor(), priority=50)
    return registry


def build_container(settings: PipelineSettings | None = None) -> Container:
    raise RuntimeError(
        "Production container cannot be built until the SQL repository adapter "
        "and its SQLAlchemy session factory are implemented. Use "
        "build_test_container() for local and test execution."
    )


def build_test_container(**overrides: object) -> Container:
    import hashlib

    from .domain.models import ContentVersion
    from .ports.language_detector import DetectionResult
    from .ports.search_index import IndexedResource, SearchHit
    from .ports.translator import TranslatedChunk

    class ResourceRecord(Protocol):
        resource_id: str
        content_hash: str
        status: object

    class DocumentRecord(Protocol):
        resource_id: str

    class JobRecord(Protocol):
        stage: str

    class Resources:
        def __init__(self) -> None:
            self.items: dict[str, ResourceRecord] = {}
        def add(self, resource: ResourceRecord) -> None:
            self.items[resource.resource_id] = resource
        def get(self, resource_id: str) -> ResourceRecord: return self.items[resource_id]
        def find_by_content_hash(self, content_hash: str) -> ResourceRecord | None:
            return next(
                (
                    resource
                    for resource in self.items.values()
                    if resource.content_hash == content_hash
                ),
                None,
            )
        def save(self, resource: ResourceRecord, *, expected_status: object) -> None:
            """Conditional update, matching the port and the real adapters.

            In-memory, so "conditional" is a status comparison rather than a
            WHERE clause. It is still worth having: a test double that accepts
            a write the database would refuse is how a lost-update bug reaches
            production behind a green suite.
            """
            existing = self.items.get(resource.resource_id)
            if existing is not None and existing.status is not expected_status:
                raise InvalidStateTransition(
                    f"Concurrent modification of resource "
                    f"{resource.resource_id!r}: expected "
                    f"{getattr(expected_status, 'value', expected_status)}, found "
                    f"{getattr(existing.status, 'value', existing.status)}."
                )
            self.items[resource.resource_id] = resource
        def list_by_status(self, status: object, *, limit: int = 100, offset: int = 0) -> list[ResourceRecord]:
            return [
                resource
                for resource in self.items.values()
                if resource.status == status
            ][offset:offset + limit]

    class Documents:
        def __init__(self) -> None: self.items: dict[str, DocumentRecord] = {}
        def save_document(self, document: DocumentRecord) -> None:
            self.items[document.resource_id] = document
        def get_document(self, resource_id: str) -> DocumentRecord: return self.items[resource_id]

    class Versions:
        def __init__(self) -> None: self.items: dict[str, list[ContentVersion]] = {}
        def save_version(self, version: ContentVersion) -> None:
            self.items.setdefault(version.resource_id, []).append(version)
        def get_latest(self, resource_id: str) -> ContentVersion | None:
            versions = self.items.get(resource_id, [])
            return versions[-1] if versions else None
        def get_machine_version(self, resource_id: str) -> ContentVersion | None:
            return next((v for v in self.items.get(resource_id, []) if str(v.author_kind).lower().endswith("machine")), None)
        def list_versions(self, resource_id: str) -> list[ContentVersion]: return list(self.items.get(resource_id, []))

    class Reviews:
        def __init__(self) -> None: self.audit: list[object] = []
        def append_audit(self, event: object) -> None: self.audit.append(event)

    class TestQueue:
        def __init__(self) -> None:
            self.queues: dict[str, list[JobRecord]] = {}
            self.dead_letter: list[tuple[JobRecord, str]] = []
        def publish(self, job: JobRecord) -> None:
            self.queues.setdefault(job.stage, []).append(job)
        def depth(self, stage: str) -> int: return len(self.queues.get(stage, []))
        def send_to_dead_letter(self, job: JobRecord, *, reason: str) -> None:
            self.dead_letter.append((job, reason))

    class TestObjectStore:
        def __init__(self) -> None: self.items: dict[str, bytes] = {}
        def put(self, key: str, content: bytes, *, content_type: str) -> str:
            self.items[key] = content
            return key
        def get(self, key: str) -> bytes: return self.items[key]
        def exists(self, key: str) -> bool: return key in self.items

    class TestSearch:
        def __init__(self) -> None: self.items: dict[str, IndexedResource] = {}
        def index(self, resource: IndexedResource) -> None: self.items[resource.resource_id] = resource
        def search(self, query: str, *, limit: int = 20, offset: int = 0) -> list[SearchHit]:
            needle = query.lower()
            matches = [
                SearchHit(resource_id=item.resource_id, title=item.title, snippet=item.translated_text, score=1.0)
                for item in self.items.values()
                if needle in item.translated_text.lower() or (item.title and needle in item.title.lower())
            ]
            return matches[offset:offset + limit]
        def remove(self, resource_id: str) -> None: self.items.pop(resource_id, None)

    class TestDeduplicator:
        def __init__(self, resources: Resources) -> None: self.resources = resources
        def compute_hash(self, *, source_url: str, content: bytes | str) -> str:
            raw = content if isinstance(content, bytes) else content.encode()
            return hashlib.sha256(source_url.encode() + b"\0" + raw).hexdigest()
        def is_duplicate(self, content_hash: str) -> bool:
            return self.resources.find_by_content_hash(content_hash) is not None

    class TestDetector:
        def __init__(self, language: str = "en", confidence: float = 0.99) -> None:
            self.language, self.confidence = language, confidence
        def detect(self, text: str) -> DetectionResult:
            return DetectionResult(self.language, self.confidence)

    class TestTranslator:
        engine_name = "test-translator"
        def supports(self, source_language: str, target_language: str) -> bool: return True
        def translate_batch(self, texts: list[str], *, source_language: str, target_language: str = "sw") -> list[TranslatedChunk]:
            return [TranslatedChunk(f"[sw] {text}") for text in texts]

    detector: object = TestDetector()
    translator: object = TestTranslator()
    if overrides.get("detector") == "low_confidence": detector = TestDetector("en", 0.5)
    elif overrides.get("detector") == "swahili": detector = TestDetector("sw")
    elif overrides.get("detector") == "english": detector = TestDetector("en")
    if overrides.get("translator") == "mismatched":
        class MismatchedTranslator(TestTranslator):
            def translate_batch(self, texts: list[str], *, source_language: str, target_language: str = "sw") -> list[TranslatedChunk]:
                return super().translate_batch(texts[:-1], source_language=source_language, target_language=target_language)
        translator = MismatchedTranslator()

    resources = Resources()
    dependencies: dict[str, object] = {
        "settings": PipelineSettings(), "queue": TestQueue(),
        "object_store": TestObjectStore(), "search": TestSearch(),
        "deduplicator": TestDeduplicator(resources),
        "detector": detector, "translator": translator,
        "fetchers": FetcherRegistry(), "extractors": ExtractorRegistry(),
        "resources": resources, "documents": Documents(), "versions": Versions(), "reviews": Reviews(),
    }
    unknown = set(overrides) - set(dependencies)
    if unknown:
        raise ValueError(f"Unknown test-container override(s): {', '.join(sorted(unknown))}")
    dependencies.update(overrides)
    dependencies["detector"] = detector if isinstance(dependencies["detector"], str) else dependencies["detector"]
    dependencies["translator"] = translator if isinstance(dependencies["translator"], str) else dependencies["translator"]
    return Container(
        settings=cast(PipelineSettings, dependencies["settings"]),
        queue=cast(JobQueue, dependencies["queue"]),
        object_store=cast(ObjectStore, dependencies["object_store"]),
        search=cast(SearchIndex, dependencies["search"]),
        deduplicator=cast(Deduplicator, dependencies["deduplicator"]),
        detector=cast(LanguageDetector, dependencies["detector"]),
        translator=cast(Translator, dependencies["translator"]),
        fetchers=cast(FetcherRegistry, dependencies["fetchers"]),
        extractors=cast(ExtractorRegistry, dependencies["extractors"]),
        resources=cast(ResourceRepository, dependencies["resources"]),
        documents=cast(DocumentRepository, dependencies["documents"]),
        versions=cast(VersionRepository, dependencies["versions"]),
        reviews=cast(ReviewRepository, dependencies["reviews"]),
    )
