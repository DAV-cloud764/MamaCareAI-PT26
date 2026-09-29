from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules.pipeline.api.routes_pipeline import router
from modules.pipeline.container import build_test_container
from modules.pipeline.domain.enums import ResourceStatus, SourceType
from modules.pipeline.domain.models import Resource


def create_test_app():
    app = FastAPI()
    container = build_test_container()

    app.state.container = container
    app.include_router(router)

    return app, container

def test_upload_pdf_successfully():
    app, container = create_test_app()
    client = TestClient(app)

    pdf_content = b"%PDF-1.4\nfake pdf content"

    response = client.post(
        "/pipeline/resources/upload",
        files={
        "file": (
            "maternal-guide.pdf",
            pdf_content,
            "application/pdf",
          )
        },
    )

    assert response.status_code == 202

    data = response.json()

    assert "resource_id" in data
    assert data["status"] == "fetched"

    resource_id = data["resource_id"]

    resource = container.resources.get(resource_id)

    assert resource.status.value == "fetched"
    assert resource.raw_object_key is not None
    assert resource.content_hash is not None

    assert container.queue.depth("extract") == 1
    assert container.object_store.exists(resource.raw_object_key)


def test_upload_rejects_non_pdf():
    app, _ = create_test_app()
    client = TestClient(app)

    response = client.post(
        "/pipeline/resources/upload",
        files={
            "file": (
                "not-pdf.txt",
                b"this is not a PDF",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_get_resource_status():
    app, container = create_test_app()
    client = TestClient(app)

    pdf_content = b"%PDF-1.4\nfake pdf content"

    upload_response = client.post(
           "/pipeline/resources/upload",
           files={
               "file": (
                   "maternal-guide.pdf",
                   pdf_content,
                   "application/pdf",
               )
           },
    )

    assert upload_response.status_code == 202

    resource_id = upload_response.json()["resource_id"]

    response = client.get(f"/pipeline/resources/{resource_id}")

    assert response.status_code == 200

    data = response.json()

    assert data["resource_id"] == resource_id
    assert data["status"] == "fetched"
    assert data["source_type"] == "pdf"
    assert data["current_version"] is None
    assert data["error"] is None

    resource = container.resources.get(resource_id)

    assert data["source_url"] == resource.source_url
    assert datetime.fromisoformat(data["submitted_at"]) == resource.submitted_at
    assert datetime.fromisoformat(data["updated_at"]) == resource.updated_at


def test_list_resources_filters_by_status():
   app, container = create_test_app()
   client = TestClient(app)

   submitted_a = Resource(
       resource_id="r-1",
       source_type=SourceType.WEB,
       source_url="https://example.com/alpha",
       status=ResourceStatus.SUBMITTED,
   )
   submitted_b = Resource(
       resource_id="r-2",
       source_type=SourceType.WEB,
       source_url="https://example.com/beta",
       status=ResourceStatus.SUBMITTED,
   )
   fetched = Resource(
       resource_id="r-3",
       source_type=SourceType.PDF,
       source_url="https://example.com/gamma",
       status=ResourceStatus.FETCHED,
   )
   container.resources.add(submitted_a)
   container.resources.add(submitted_b)
   container.resources.add(fetched)

   response = client.get("/pipeline/resources", params={"status": "submitted"})

   assert response.status_code == 200
   assert [item["resource_id"] for item in response.json()] == ["r-1", "r-2"]


def test_list_resources_uses_limit_and_offset():
   app, container = create_test_app()
   client = TestClient(app)

   for index in range(5):
       container.resources.add(
           Resource(
               resource_id=f"r-{index}",
               source_type=SourceType.WEB,
               source_url=f"https://example.com/{index}",
               status=ResourceStatus.SUBMITTED,
           )
       )

   response = client.get(
       "/pipeline/resources",
       params={"status": "submitted", "limit": 2, "offset": 1},
   )

   assert response.status_code == 200
   assert [item["resource_id"] for item in response.json()] == ["r-1", "r-2"]


def test_list_resources_caps_limit_at_500():
   app, container = create_test_app()
   client = TestClient(app)

   for index in range(600):
       container.resources.add(
           Resource(
               resource_id=f"r-{index}",
               source_type=SourceType.WEB,
               source_url=f"https://example.com/{index}",
               status=ResourceStatus.SUBMITTED,
           )
       )

   response = client.get(
       "/pipeline/resources",
       params={"status": "submitted", "limit": 600},
   )

   assert response.status_code == 200
   assert len(response.json()) == 500


def test_list_resources_rejects_invalid_pagination():
   app, _ = create_test_app()
   client = TestClient(app)

   response = client.get(
       "/pipeline/resources",
       params={"status": "submitted", "limit": 0},
   )
   assert response.status_code == 422

   response = client.get(
       "/pipeline/resources",
       params={"status": "submitted", "offset": -1},
   )
   assert response.status_code == 422


def test_get_stats_returns_queue_and_status_counts():
   app, container = create_test_app()
   client = TestClient(app)

   container.queue.publish(type("Job", (), {"stage": "ingest"})())
   container.queue.publish(type("Job", (), {"stage": "ingest"})())
   container.queue.publish(type("Job", (), {"stage": "review"})())

   container.resources.add(
       Resource(
           resource_id="stats-1",
           source_type=SourceType.WEB,
           source_url="https://example.com/s1",
           status=ResourceStatus.SUBMITTED,
       )
   )
   container.resources.add(
       Resource(
           resource_id="stats-2",
           source_type=SourceType.PDF,
           source_url="https://example.com/s2",
           status=ResourceStatus.FETCHED,
       )
   )
   container.resources.add(
       Resource(
           resource_id="stats-3",
           source_type=SourceType.WEB,
           source_url="https://example.com/s3",
           status=ResourceStatus.FETCHED,
       )
   )

   response = client.get("/pipeline/stats")

   assert response.status_code == 200
   data = response.json()
   assert "queue_depth" in data
   assert "resource_counts" in data
   assert data["queue_depth"]["ingest"] == 2
   assert data["queue_depth"]["review"] == 1
   assert data["resource_counts"]["submitted"] == 1
   assert data["resource_counts"]["fetched"] == 2
   assert "oldest_review_age_seconds" not in data
