# MamaCareAI pipeline review, React UI, and deployment plan

This is the file ownership plan for a deployable MamaCareAI release. The frontend is a separate React + TypeScript (Vite) application. FastAPI serves the API; a separate worker runs the asynchronous pipeline. PostgreSQL stores structured records and versions; persistent object storage holds raw fetched content. A durable queue carries jobs. These services can be deployed independently.

**Ownership:** One developer owns each file. Paths marked *new* are proposed files; adapt an existing equivalent in the repository instead of creating duplicate infrastructure. Agree on API schemas, status values, and deployment environment names before parallel work. No secrets belong in the frontend. Do not treat completion of a checklist as proof of deployability: run the acceptance checks at the end.

| Developer | File to touch | What to implement in that file |
| --- | --- | --- |
| Dev A | `backend/modules/pipeline/services/submission.py` | Normalize URLs before duplicate checks, enforce the vetted-source rule, persist the submitted resource and enqueue its job; return promptly without fetching in the HTTP request. |
| Dev A | `backend/modules/pipeline/stages/ingest.py` | Separate URL and content deduplication; save raw fetch metadata and object key; record retryable/permanent failures. |
| Dev A | `backend/modules/pipeline/stages/extract.py` | Reject noncontiguous block order and unusable extraction; save the original ordered `NormalizedDocument` unchanged. |
| Dev A | `backend/modules/pipeline/adapters/fetchers/web_fetcher.py` | Limit schemes, redirects, timeouts and bytes; validate DNS/IP for initial and redirected hosts to reject private or local destinations. |
| Dev A | `backend/modules/pipeline/adapters/extractors/html_extractor.py` | Filter navigation, cookie banners, footers and repeated text while preserving useful article blocks and ordering. |
| Dev A | `backend/modules/pipeline/api/routes_pipeline.py` | Provide authenticated URL submission, paginated resource listing and resource status/detail endpoints; expose safe error codes and stage progress for React polling. |
| Dev A | `backend/tests/pipeline/test_web_pipeline.py` *new* | Cover URL safety, redirects, size limits, duplicates, extraction quality, ordering and original-text preservation. |
| Dev B | `backend/modules/pipeline/config.py` | Validate environment settings for selected translator, provider credentials, language policy, queue, object store and database; fail startup clearly on missing production settings. |
| Dev B | `backend/modules/pipeline/ports/translator.py` | Keep a provider-neutral chunk translation interface and normalized error/result types. |
| Dev B | `backend/modules/pipeline/stages/detect_language.py` | Route English to translation and Swahili to direct review; route other or uncertain languages to explicit manual/unsupported status. |
| Dev B | `backend/modules/pipeline/stages/translate.py` | Translate through the port, validate nonempty aligned output, and create a machine version without provider-specific imports. |
| Dev B | `backend/modules/pipeline/adapters/translation/gemini_translator.py` | Map API responses, timeouts and rate limits into the shared translation contract. |
| Dev B | `backend/modules/pipeline/adapters/translation/cloud_translator.py` | Normalize each actually supported cloud provider behind the same contract; document providers that remain unimplemented rather than advertising them as deployable. |
| Dev B | `backend/modules/pipeline/container.py` | Replace the unfinished `build_container()` path with production wiring for SQL repositories, persistent queue, persistent object storage, fetch/extract, detector, translator and stages. Keep web and worker processes on the same configuration. |
| Dev B | `backend/modules/pipeline/worker.py` *new or existing entry point* | Start a separate long-running worker; consume/ack jobs only after durable processing, apply bounded retries and dead-letter/failed states, and resume safely after restart. |
| Dev B | `backend/config/.env.example` | List server-only database, queue, object-store and translation variables and required origins; use placeholders without real credentials. |
| Dev B | `backend/tests/pipeline/test_translation_pipeline.py` *new* | Cover language routing, provider switching, malformed output and transient/permanent failures. |
| Dev B | `backend/tests/pipeline/test_worker_recovery.py` *new* | Verify retry limits, restart/reprocessing safety and no lost submitted jobs against the chosen queue adapter. |
| Dev C | `backend/modules/pipeline/domain/models.py` | Define canonical immutable extracted documents and distinct machine/human `ContentVersion` records. |
| Dev C | `backend/modules/pipeline/domain/enums.py` | Define stable processing, confirmation, review, failure, approval and publication statuses used by API and frontend. |
| Dev C | `backend/modules/pipeline/ports/repositories.py` | Specify contracts for resource, original document, version, assignment and audit persistence, including atomic version checks. |
| Dev C | `backend/modules/pipeline/adapters/storage/sql_repositories.py` | Remove duplicate domain definitions, map canonical models to PostgreSQL, persist originals and append-only versions separately, and enforce concurrency/transaction boundaries. |
| Dev C | `backend/modules/pipeline/adapters/storage/repository_factory.py` *new* | Construct SQLAlchemy sessions and repositories from the database URL and expose a clear contract to `container.py`. |
| Dev C | `backend/alembic/versions/<pipeline_schema_revision>.py` *new* | Add a reviewed migration for resources, documents/blocks, versions, assignments and audit records, with keys, indexes, uniqueness and foreign keys; never depend on runtime `create_all()` in production. |
| Dev C | `backend/modules/pipeline/stages/store.py` | Create a reviewable source version for native Swahili, keep originals untouched and prevent unapproved content from entering the production knowledge index. |
| Dev C | `backend/modules/pipeline/services/review_service.py` | Validate reviewer identity and assignments; append edits as new versions, accept no-change approval, and reject stale `base_version_number` writes. |
| Dev C | `backend/modules/pipeline/stages/publish.py` | Publish only explicitly approved versions; point the knowledge index to the approved content and keep a durable audit trail. |
| Dev C | `backend/modules/pipeline/api/schemas.py` | Freeze typed request/response shapes for list, status, source blocks, aligned review units, history, edits and decisions; include stable error codes and version number. |
| Dev C | `backend/modules/pipeline/api/routes_review.py` | Expose authenticated review detail/history/edit/decision endpoints; enforce roles, map conflicts to `409`, and return appropriate validation/auth errors. |
| Dev C | `backend/tests/pipeline/test_review_persistence.py` *new* | Run PostgreSQL-backed tests for originals, machine/human versions, Swahili review, stale edits, authorization and publish gating. |
| Dev D | `frontend/package.json` *new* | Define Vite, React, TypeScript, build/check/test scripts and pinned dependencies. |
| Dev D | `frontend/index.html` *new* | Provide the HTML mounting point and application metadata; do not embed provider keys. |
| Dev D | `frontend/src/main.tsx` *new* | Mount React, router and shared data/auth providers. |
| Dev D | `frontend/src/App.tsx` *new* | Define navigation and protected routes for ingestion, documents and review pages. |
| Dev D | `frontend/src/api/client.ts` *new* | Set the API base URL from `VITE_API_BASE_URL`, send same-site cookies or approved auth credentials, normalize errors, and avoid storing secrets in browser code. |
| Dev D | `frontend/src/api/pipeline.ts` *new* | Implement typed submit/list/status API calls that match Dev A/C contracts. |
| Dev D | `frontend/src/pages/IngestionPage.tsx` *new* | Build the accessible URL form and polling status view with loading, retry, error and terminal/waiting states. |
| Dev D | `frontend/src/styles.css` *new* | Define shared responsive styles, readable status labels and keyboard focus treatment. |
| Dev D | `frontend/.env.example` *new* | Document the public API base URL only; clarify that all `VITE_` values are visible to browsers. |
| Dev D | `frontend/vite.config.ts` *new* | Configure development API proxy and build output; production requests target the deployed API origin. |
| Dev D | `frontend/src/pages/IngestionPage.test.tsx` *new* | Test submit, polling termination and error/empty states with mocked API responses. |
| Dev E | `frontend/src/api/review.ts` *new* | Define typed list/detail/history/edit/decision calls using Dev C schemas; submit `base_version_number` and preserve error status `409`. |
| Dev E | `frontend/src/pages/DocumentsPage.tsx` *new* | Show paginated resources, status filters, source metadata, loading/empty/error states and review navigation. |
| Dev E | `frontend/src/pages/ReviewPage.tsx` *new* | Display immutable original blocks beside aligned editable Swahili units; allow notes, save, history and authorized review decisions; distinguish machine versus reviewed text. |
| Dev E | `frontend/src/components/ReviewEditor.tsx` *new* | Keep unit IDs/order and Unicode intact; track unsaved changes; show explicit save and stale-version conflict/reload behavior. |
| Dev E | `frontend/src/components/VersionHistory.tsx` *new* | Show provenance, engine/reviewer, time and notes without altering earlier versions. |
| Dev E | `frontend/src/pages/ReviewPage.test.tsx` *new* | Test aligned display, edits, no-change approval, unsaved warning, conflict behavior and authorization-dependent controls. |

## Deployment contracts and release checks

The web UI deploys as static frontend assets (or a static hosting service). FastAPI deploys as an API service and the worker as a separate process from the same backend image. Both backend processes connect to the same PostgreSQL database, durable queue and persistent object store; only the backend uses translation credentials. If another deployment shape is chosen, preserve these persistence and process boundaries.

| Contract or gate | Acceptance evidence |
| --- | --- |
| API and browser | Published API base URL and allowed frontend origin configured; authenticated requests work across origins with appropriate CORS/CSRF/session settings; ingestion and review endpoints reject unauthorized callers. |
| Database | Apply Alembic migrations to a fresh PostgreSQL instance and on an upgrade path; restart API/worker and retrieve originals, versions and review state. Backups and restore procedure are exercised. |
| Object storage | Configured persistent bucket and least-privilege backend access; upload and retrieve raw content after a worker restart. The database stores object references, not local ephemeral paths. |
| Queue and worker | Real durable queue adapter selected for production, jobs survive a restart, retry limits are enforced and failed jobs remain inspectable. In-memory queue is development only. |
| Secrets and deployment | Translation keys, database credentials and object-store credentials are server-side environment secrets; health/readiness checks and worker logs are available. Frontend build receives only public settings. |
| English end to end | Submit vetted English URL → save raw object and original English blocks → translate with selected provider → save machine Swahili version → human edit creates another version → approve → publish approved version. All prior content remains retrievable. |
| Swahili end to end | Submit vetted Swahili URL → save original → skip translation → create review version → approve (with or without edit) → publish. |
| Safety and failure | Unsupported language awaits manual action; translation outage preserves original and produces a bounded retry/failure; stale edit returns `409`; unpublished content is absent from the production knowledge index. |
| Browser smoke test | From the deployed React URL, log in, ingest, monitor status, list/open a document, compare source and review text, save an edit, and approve with an authorized account. |

**Integration order:** Dev C publishes API schemas and PostgreSQL repository contract; Dev A/B connect pipeline and worker; Dev D/E develop against typed mock responses, then connect to the live API. Merge only after the full environment passes these checks. Infrastructure manifests for the selected host (database/queue/bucket provisioning, service commands and environment configuration) must be added and owned in a separate deployment PR once the hosting provider is chosen; this document does not claim those manifests exist.
