"""
Operator CLI — the tools you will wish you had at 11pm before a demo.

    python -m backend.modules.pipeline.cli submit --url https://... --type pdf
    python -m backend.modules.pipeline.cli status --resource-id <id>
    python -m backend.modules.pipeline.cli requeue --stage translate --failed
    python -m backend.modules.pipeline.cli reindex
    python -m backend.modules.pipeline.cli export-feedback --since 2026-01-01

BUILD THESE EARLY, NOT WHEN YOU NEED THEM.
Every one of these commands exists because a real pipeline hits a day where a
provider outage dead-letters 200 documents, or the search index drifts from the
database. Without `requeue` and `reindex`, the recovery is hand-written SQL at
midnight. With them, it is one command that has already been tested.

They are thin: each command calls a service and prints the result. No business
logic here — everything is reachable from the API and the CLI because it lives
in `services/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime

STAGES = (
    "ingest",
    "extract",
    "detect_language",
    "translate",
    "store",
    "review",
    "publish",
)


def _build_container_for_cli():
    """Build container, falling back to test container until SQL repos ready."""
    try:
        from .config import PipelineSettings
        from .container import build_container

        try:
            settings = PipelineSettings()
            return build_container(settings)
        except RuntimeError:
            from .container import build_test_container

            return build_test_container()
    except Exception as exc:  # noqa: BLE001
        print(f"failed to build container: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_submit(args: argparse.Namespace) -> int:
    container = _build_container_for_cli()
    # Use SubmissionService if available, otherwise direct resource creation for test containers
    try:
        from .domain.enums import SourceType
        from .services.submission import SubmissionService

        # Build source_register that always approves for CLI test mode
        class _AlwaysApproved:
            def is_approved(self, url: str) -> bool:
                return True

        service = SubmissionService(
            resources=container.resources,
            queue=container.queue,
            source_register=_AlwaysApproved(),
        )
        source_type = None
        if args.type:
            try:
                source_type = SourceType(args.type)
            except ValueError:
                print(f"unsupported source type: {args.type}", file=sys.stderr)
                return 1
        metadata = None
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except json.JSONDecodeError as exc:
                print(f"invalid --metadata JSON: {exc}", file=sys.stderr)
                return 1
        resource = service.submit(
            source_url=args.url,
            source_type=source_type,
            submitted_by="cli",
            metadata=metadata,
        )
        print(resource.resource_id)
        return 0
    except Exception as exc:  # noqa: BLE001
        # Fallback for test containers where SubmissionService may be stub
        print(f"submit failed: {exc}", file=sys.stderr)
        return 1


def _cmd_process_url(args: argparse.Namespace) -> int:
    """Synchronously fetch, extract, and conditionally translate one URL.

    This command is intentionally independent of the queued pipeline so it is
    useful for local, one-off ingestion and produces the final text on stdout.
    English content is translated to the configured target language (Swahili
    by default); content in any other detected language is returned unchanged.
    """
    try:
        from .adapters.translation.chunker import Chunker
        from .config import PipelineSettings
        from .container import (
            build_detector,
            build_extractors,
            build_fetchers,
            build_translator,
        )
        from .domain.enums import SourceType
        from .services.submission import SubmissionService

        settings = PipelineSettings()
        SubmissionService._validate_url(args.url)
        source_type = (
            SourceType(args.type)
            if args.type
            else SubmissionService._infer_source_type(args.url)
        )

        fetch_result = build_fetchers(settings).get(source_type).fetch(args.url)
        extractor = build_extractors(settings).select(
            fetch_result.content_type, fetch_result.content
        )
        document = extractor.extract(
            "cli-process-url",
            fetch_result.content,
            metadata={
                **fetch_result.metadata,
                "content_type": fetch_result.content_type,
            },
        )
        extracted_text = "\n\n".join(
            block.text for block in document.blocks if block.text.strip()
        )
        if not extracted_text:
            raise ValueError("extractor returned no usable text")

        detection = build_detector(settings).detect(extracted_text)
        if detection.language.casefold() != "en":
            print(f"=== ORIGINAL ({detection.language}) ===")
            print(extracted_text)
            print("\n=== TRANSLATED (sw) ===")
            print("Translation skipped: detected language is not English.")
            return 0

        translator = build_translator(settings)
        if not translator.supports("en", settings.target_language):
            raise ValueError(
                f"{translator.engine_name} does not support English to "
                f"{settings.target_language} translation"
            )
        chunks = Chunker(max_chars=settings.translation_max_chunk_chars).chunk(
            document.blocks
        )
        translated = translator.translate_batch(
            [chunk.text for chunk in chunks],
            source_language="en",
            target_language=settings.target_language,
        )
        if len(translated) != len(chunks):
            raise ValueError(
                "translator returned a different number of chunks than it received"
            )
        translated_text = "\n\n".join(chunk.text for chunk in translated)
        print("=== ORIGINAL (en) ===")
        print(extracted_text)
        print(f"\n=== TRANSLATED ({settings.target_language}) ===")
        print(translated_text)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports clean errors
        print(f"process-url failed: {exc}", file=sys.stderr)
        return 1


def _cmd_status(args: argparse.Namespace) -> int:
    container = _build_container_for_cli()
    rid = args.resource_id
    try:
        resource = container.resources.get(rid)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        # Try find in any storage
        resources = getattr(container.resources, "items", None)
        if resources is not None and rid in resources:
            resource = resources[rid]
        else:
            print(f"resource {rid} not found", file=sys.stderr)
            return 1

    # Basic resource info
    out = {
        "resource_id": getattr(resource, "resource_id", rid),
        "source_url": getattr(resource, "source_url", None),
        "source_type": str(getattr(resource, "source_type", "")),
        "status": str(getattr(resource, "status", "")),
        "attempt_count": getattr(resource, "attempt_count", None),
        "last_error": getattr(resource, "last_error", None),
        "detected_language": getattr(resource, "detected_language", None),
        "language_confidence": getattr(resource, "language_confidence", None),
        "submitted_at": str(getattr(resource, "submitted_at", "")),
        "updated_at": str(getattr(resource, "updated_at", "")),
    }

    # Version history
    try:
        versions = container.versions.list_versions(rid)  # type: ignore[attr-defined]
        out["versions"] = [
            {
                "version_number": v.version_number,
                "author_kind": str(v.author_kind),
                "engine": getattr(v, "engine", None),
                "created_at": str(getattr(v, "created_at", "")),
                "note": getattr(v, "note", None),
            }
            for v in versions
        ]
    except Exception:  # noqa: BLE001
        out["versions"] = []

    # Audit if available
    try:
        audit = container.reviews.list_audit(rid)  # type: ignore[attr-defined]
        out["audit"] = [str(a) for a in audit[:10]]
    except Exception:  # noqa: BLE001, S110
        pass

    print(json.dumps(out, indent=2, default=str))
    return 0


def _cmd_not_implemented(args: argparse.Namespace) -> int:
    print("not implemented — see PIPE-34 (export-feedback)", file=sys.stderr)
    return 2


def _dead_letter_entries(queue: object) -> list:
    """Best-effort read of dead-letter entries across queue implementations."""
    for attr in ("_dead_letter", "dead_letter"):
        entries = getattr(queue, attr, None)
        if isinstance(entries, list):
            return entries
    return []


def _dead_letter_count(queue: object) -> int:
    return len(_dead_letter_entries(queue))


def _queue_depth(queue: object, stage: str) -> int:
    depth = getattr(queue, "depth", None)
    if callable(depth):
        try:
            return int(depth(stage))  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return 0
    return 0


def _list_by_status(resources: object, status: object) -> list:
    lister = getattr(resources, "list_by_status", None)
    if not callable(lister):
        return []
    try:
        return list(lister(status, limit=10000))  # type: ignore[arg-type]
    except TypeError:
        try:
            return list(lister(status))  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return []
    except Exception:  # noqa: BLE001
        return []


def _all_resources(resources: object) -> list:
    """Aggregate resources across all statuses (port has no list-all)."""
    from .domain.enums import ResourceStatus

    seen: dict[str, object] = {}
    for status in ResourceStatus:
        for resource in _list_by_status(resources, status):
            rid = getattr(resource, "resource_id", None)
            if rid is not None:
                seen[rid] = resource
    if not seen:
        items = getattr(resources, "items", None)
        if isinstance(items, dict):
            return list(items.values())
    return list(seen.values())


def _republish_dead_letters(container: object, *, stage: str | None) -> int:
    """Republish dead-letter jobs as new PENDING jobs. Returns count."""
    from .domain.models import Job, JobStatus

    queue = container.queue  # type: ignore[attr-defined]
    entries = list(_dead_letter_entries(queue))
    republished = 0
    remaining: list = []
    for entry in entries:
        job = entry[0] if isinstance(entry, (tuple, list)) else entry
        job_stage = getattr(job, "stage", None)
        if stage is not None and job_stage != stage:
            remaining.append(entry)
            continue
        try:
            queue.publish(  # type: ignore[attr-defined]
                Job(
                    job_id=str(uuid.uuid4()),
                    resource_id=job.resource_id,
                    stage=job.stage,
                    status=JobStatus.PENDING,
                    attempts=0,
                )
            )
            republished += 1
        except Exception:  # noqa: BLE001
            remaining.append(entry)
    # Best-effort removal so a second requeue is a no-op. No port method
    # exists for this; mutate the concrete list in place when visible.
    for attr in ("_dead_letter", "dead_letter"):
        store = getattr(queue, attr, None)
        if isinstance(store, list):
            try:
                store[:] = remaining
            except Exception:  # noqa: BLE001, S110
                pass
            break
    return republished


def _retry_failed_resources(
    container: object, *, resource_id: str | None
) -> int:
    """Reset FAILED resources to SUBMITTED and re-enqueue at ingest.

    Full restart (not stage-targeted resume) is the only resume that is
    always correct without artifact-existence checks: ingest short-circuits
    via raw_object_key when raw bytes already exist, and downstream stages
    rebuild from there. Stage-targeted resume is future work.
    """
    from .domain.enums import ResourceStatus
    from .domain.models import Job, JobStatus

    resources = container.resources  # type: ignore[attr-defined]
    queue = container.queue  # type: ignore[attr-defined]

    if resource_id is not None:
        try:
            candidates = [resources.get(resource_id)]  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            print(f"resource {resource_id} not found", file=sys.stderr)
            return 0
    else:
        candidates = _list_by_status(resources, ResourceStatus.FAILED)

    retried = 0
    for resource in candidates:
        if getattr(resource, "status", None) != ResourceStatus.FAILED:
            continue
        try:
            reset = resource.with_status(  # type: ignore[attr-defined]
                ResourceStatus.SUBMITTED, attempt_count=0, last_error=None
            )
            resources.save(  # type: ignore[attr-defined]
                reset, expected_status=resource.status
            )
            queue.publish(  # type: ignore[attr-defined]
                Job(
                    job_id=str(uuid.uuid4()),
                    resource_id=resource.resource_id,
                    stage="ingest",
                    status=JobStatus.PENDING,
                    attempts=0,
                )
            )
            retried += 1
        except Exception as exc:  # noqa: BLE001
            print(f"requeue failed for {resource.resource_id}: {exc}", file=sys.stderr)
    return retried


def _cmd_requeue(args: argparse.Namespace) -> int:
    container = _build_container_for_cli()
    if args.stage is not None and args.stage not in STAGES:
        print(
            f"unknown stage {args.stage!r}. Valid stages: {', '.join(STAGES)}",
            file=sys.stderr,
        )
        return 1
    if args.resource_id is not None:
        dl_count = _republish_dead_letters(container, stage=args.stage)
        # If the resource itself is FAILED, restart it at ingest too.
        retried = _retry_failed_resources(container, resource_id=args.resource_id)
        out: dict[str, object] = {
            "requeued": dl_count + retried,
            "resource_id": args.resource_id,
        }
        if args.stage is not None:
            out["stage"] = args.stage
        print(json.dumps(out, default=str))
        return 0
    if args.failed:
        retried = _retry_failed_resources(container, resource_id=None)
        print(json.dumps({"requeued": retried, "restart_at": "ingest"}, default=str))
        return 0
    count = _republish_dead_letters(container, stage=args.stage)
    out = {"requeued": count}
    if args.stage is not None:
        out["stage"] = args.stage  # type: ignore[dict-item]
    print(json.dumps(out, default=str))
    return 0


def _cmd_reindex(args: argparse.Namespace) -> int:
    container = _build_container_for_cli()
    from .ports.search_index import IndexedResource

    since = None
    if args.since is not None:
        try:
            since = datetime.fromisoformat(args.since)
            if since.tzinfo is None:
                since = since.replace(tzinfo=UTC)
        except ValueError:
            print(f"invalid --since {args.since!r}, expected ISO date", file=sys.stderr)
            return 1

    indexed = 0
    skipped = 0
    for resource in _all_resources(container.resources):  # type: ignore[attr-defined]
        rid = getattr(resource, "resource_id", None)
        try:
            version = container.versions.get_latest(rid)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            version = None
        if version is None:
            skipped += 1
            continue
        if since is not None:
            created = getattr(version, "created_at", None)
            try:
                if created is not None and created < since:
                    skipped += 1
                    continue
            except TypeError:
                pass
        try:
            document = container.documents.get_document(rid)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        units = getattr(version, "units", ()) or ()
        text = "\n\n".join(getattr(u, "translated_text", "") for u in units)
        status_obj: object = getattr(resource, "status", "")
        if rid is None:
            skipped += 1
            continue
        status_str = getattr(status_obj, "value", str(status_obj))
        try:
            container.search.index(  # type: ignore[attr-defined]
                IndexedResource(
                    resource_id=str(rid),
                    title=getattr(document, "title", None),
                    translated_text=text,
                    source_url=getattr(resource, "source_url", ""),
                    status=str(status_str),
                    version_number=getattr(version, "version_number", 1),
                    metadata={"engine": getattr(version, "engine", None)},
                )
            )
            indexed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"reindex failed for {rid}: {exc}", file=sys.stderr)
            skipped += 1
    print(json.dumps({"indexed": indexed, "skipped": skipped}, default=str))
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    container = _build_container_for_cli()
    from .domain.enums import ResourceStatus

    queue = container.queue  # type: ignore[attr-defined]
    resources = container.resources  # type: ignore[attr-defined]
    out = {
        "queues": {stage: _queue_depth(queue, stage) for stage in STAGES},
        "dead_letter": _dead_letter_count(queue),
        "resources": {
            status.value: len(_list_by_status(resources, status))
            for status in ResourceStatus
        },
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch a pipeline management command."""
    parser = argparse.ArgumentParser(prog="pipeline-cli", description="Pipeline operator CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # submit
    p_submit = sub.add_parser("submit", help="Submit a URL for processing")
    p_submit.add_argument("--url", required=True, help="Source URL")
    p_submit.add_argument("--type", choices=["web", "pdf", "video"], default=None, help="Source type (inferred if not given)")
    p_submit.add_argument("--metadata", default=None, help="JSON metadata string")
    p_submit.set_defaults(func=_cmd_submit)

    # process-url: synchronous one-command fetch -> extract -> translate
    p_process = sub.add_parser(
        "process-url",
        help="Fetch and extract a URL, translating English content to Swahili",
    )
    p_process.add_argument("--url", required=True, help="Source URL")
    p_process.add_argument(
        "--type",
        choices=["web", "pdf", "video"],
        default=None,
        help="Source type (inferred if not given)",
    )
    p_process.set_defaults(func=_cmd_process_url)

    # status
    p_status = sub.add_parser("status", help="Show resource status")
    p_status.add_argument("--resource-id", required=True, help="Resource ID")
    p_status.set_defaults(func=_cmd_status)

    # requeue (PIPE-30)
    p_requeue = sub.add_parser(
        "requeue",
        help="Re-drive dead-lettered jobs or restart FAILED resources at ingest (PIPE-30)",
    )
    p_requeue.add_argument("--stage", default=None, help="Stage to requeue DL jobs for")
    p_requeue.add_argument(
        "--failed",
        action="store_true",
        help="Restart FAILED resources at ingest instead of DL jobs",
    )
    p_requeue.add_argument("--resource-id", default=None)
    p_requeue.set_defaults(func=_cmd_requeue)

    # reindex (PIPE-30)
    p_reindex = sub.add_parser("reindex", help="Rebuild search index (PIPE-30)")
    p_reindex.add_argument("--since", default=None, help="ISO date, only newer versions")
    p_reindex.set_defaults(func=_cmd_reindex)

    # export-feedback (PIPE-34, deferred until >=50 human edits)
    p_export = sub.add_parser("export-feedback", help="Export feedback pairs (PIPE-34)")
    p_export.add_argument("--since", required=True)
    p_export.add_argument("--out", default=None)
    p_export.set_defaults(func=_cmd_not_implemented)

    # stats (PIPE-30)
    p_stats = sub.add_parser("stats", help="Queue depth and counts (PIPE-30)")
    p_stats.set_defaults(func=_cmd_stats)

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2

    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
