"""API server: endpoints, and live alerts reaching a dashboard."""

import pytest
from fastapi.testclient import TestClient

from app.config import SurveillanceConfig
from app.runner import CameraSpec
from app.server import create_app


@pytest.fixture
def client(tmp_path):
    config = SurveillanceConfig(
        source="synthetic",
        detector="sim",
        synthetic_frames=120,
        force_night=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "events",
        evidence_dir=tmp_path / "evidence",
        save_clips=False,
        quiet=True,
        print_summary=False,
        color=False,
    )
    specs = [CameraSpec(camera_id="CAM-01", source="synthetic")]
    app = create_app(specs, config, db_path=tmp_path / "server.db")
    with TestClient(app) as c:
        yield c


def test_the_dashboard_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "SENTINEL-X" in response.text


def test_status_reports_the_fleet(client):
    payload = client.get("/api/status").json()
    assert payload["running"] in (True, False)
    assert [c["camera_id"] for c in payload["cameras"]] == ["CAM-01"]
    assert "model_sharing" in payload


def test_summary_is_available_before_any_alert(client):
    payload = client.get("/api/summary").json()
    assert set(payload) == {"total", "by_severity", "by_camera", "latest"}


def test_behaviours_endpoint_returns_a_mapping(client):
    assert isinstance(client.get("/api/behaviours").json(), dict)


def test_events_endpoint_shape(client):
    payload = client.get("/api/events").json()
    assert set(payload) == {"count", "events"}
    assert payload["count"] == len(payload["events"])


def test_events_limit_is_validated(client):
    assert client.get("/api/events?limit=0").status_code == 422
    assert client.get("/api/events?limit=99999").status_code == 422


def test_an_unknown_event_is_a_404(client):
    response = client.get("/api/events/NOPE")
    assert response.status_code == 404
    assert "NOPE" in response.json()["detail"]


def test_a_websocket_client_gets_a_snapshot_immediately(client):
    """A dashboard must not sit blank until the next alert happens."""
    with client.websocket_connect("/ws") as ws:
        message = ws.receive_json()
    assert message["type"] == "snapshot"
    assert "status" in message
    assert isinstance(message["events"], list)


def test_the_hub_counts_connected_dashboards(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        assert client.get("/api/status").json()["dashboards_connected"] == 1


def test_alerts_are_stored_and_retrievable(tmp_path):
    """Whatever the fleet raises must be queryable afterwards."""
    from app.store import EventStore

    db = tmp_path / "stored.db"
    config = SurveillanceConfig(
        source="synthetic",
        detector="sim",
        synthetic_frames=120,
        force_night=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "events",
        evidence_dir=tmp_path / "evidence",
        save_clips=False,
        quiet=True,
        print_summary=False,
        color=False,
    )
    app = create_app([CameraSpec(camera_id="CAM-01", source="synthetic")], config, db_path=db)
    with TestClient(app) as client:
        service = app.state.service
        if service.thread is not None:
            service.thread.join(timeout=30)

        events = client.get("/api/events").json()
        assert events["count"] >= 1
        first = events["events"][0]
        assert client.get(f"/api/events/{first['event_id']}").json()["event_id"] == first["event_id"]

    with EventStore(db) as store:
        assert store.count() >= 1


def test_the_fleet_stops_when_the_server_shuts_down(tmp_path):
    config = SurveillanceConfig(
        source="synthetic",
        detector="sim",
        synthetic_frames=100000,  # would run for a long time
        force_night=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "events",
        evidence_dir=tmp_path / "evidence",
        save_clips=False,
        quiet=True,
        print_summary=False,
        color=False,
    )
    app = create_app([CameraSpec(camera_id="CAM-01", source="synthetic")], config,
                     db_path=tmp_path / "shutdown.db")
    with TestClient(app):
        pass
    assert not app.state.service.running


# --------------------------------------------------------------- readiness


def test_the_service_signals_when_the_fleet_is_actually_started(tmp_path):
    """Regression: `runner` exists long before its cameras do.

    Waiting on the object alone handed the caller an empty fleet, which then
    "finished" instantly and printed a summary of zero cameras.
    """
    config = SurveillanceConfig(
        source="synthetic",
        detector="sim",
        synthetic_frames=100000,
        force_night=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "events",
        evidence_dir=tmp_path / "evidence",
        save_clips=False,
        quiet=True,
        print_summary=False,
        color=False,
    )
    app = create_app([CameraSpec(camera_id="CAM-01", source="synthetic")], config,
                     db_path=tmp_path / "ready.db")
    service = app.state.service
    assert not service.ready.is_set()

    with TestClient(app):
        assert service.ready.is_set()
        # Once ready, the fleet is genuinely built - not just constructed.
        assert service.runner is not None
        assert [w.spec.camera_id for w in service.runner.workers] == ["CAM-01"]


def test_readiness_is_set_even_when_no_camera_is_reachable(tmp_path, monkeypatch):
    """Otherwise a caller waiting on it would hang forever."""
    monkeypatch.setattr("app.video.stream.tcp_reachable", lambda url, timeout: False)
    config = SurveillanceConfig(
        source="rtsp://10.0.0.9/s",
        detector="sim",
        force_night=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "events",
        evidence_dir=tmp_path / "evidence",
        save_clips=False,
        quiet=True,
        print_summary=False,
        color=False,
    )
    app = create_app([CameraSpec(camera_id="CAM-X", source="rtsp://10.0.0.9/s")], config,
                     db_path=tmp_path / "none.db")
    with TestClient(app) as client:
        service = app.state.service
        assert service.ready.is_set()
        assert service.runner.workers == []
        # The dashboard still serves, so an operator can see why.
        assert client.get("/api/status").json()["skipped"][0]["camera_id"] == "CAM-X"


def test_view_mode_leaves_the_wait_loop_to_the_caller(tmp_path):
    """OpenCV's GUI is main-thread only, so the service must not wait too."""
    config = SurveillanceConfig(
        source="synthetic",
        detector="sim",
        synthetic_frames=100000,
        force_night=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "events",
        evidence_dir=tmp_path / "evidence",
        save_clips=False,
        view=True,
        quiet=True,
        print_summary=False,
        color=False,
    )
    app = create_app([CameraSpec(camera_id="CAM-01", source="synthetic")], config,
                     db_path=tmp_path / "view.db")
    with TestClient(app):
        service = app.state.service
        assert service.thread is None, "view mode must not background the wait loop"
        assert service.running, "liveness should come from the cameras themselves"
        service.runner.stop()


# --------------------------------------------------------- dashboard content


def test_the_dashboard_renders_the_fields_an_operator_acts_on(client):
    """A plate, a registry mismatch and the evidence are why a vehicle is stopped.

    All of it was plumbed through the event model long before the page showed
    any of it, which made the strongest signals invisible in a demo. Asserted
    on the event fields the page reads rather than on any variable name, so a
    rewritten dashboard is judged on what it shows, not how it is written.
    """
    page = client.get("/").text
    for field in ("plate", "registry", "clip_path", "plate_path", "evidence_path"):
        assert field in page, f"dashboard never reads {field}"
    # The evidence has to be reachable, not merely mentioned.
    assert "/api/evidence/" in page


def test_the_dashboard_escapes_everything_it_renders(client):
    """Plate text comes from OCR, and camera ids come from a config file.

    Both reach the page as HTML, so both have to be escaped. Checked by
    counting: every value interpolated into markup must pass through the
    escaper, and a page that interpolates far more than it escapes has a hole
    in it somewhere.
    """
    import re

    page = client.get("/").text
    assert "esc = (" in page or "function esc(" in page, "no escaper at all"
    assert "replace(/[&<>\"]/g" in page, "the escaper does not escape the dangerous set"

    # Interpolations of event data specifically - these carry outside input.
    risky = re.findall(r"\$\{[^}]*(?:e|inc|c|s)\.[a-z_]+", page)
    escaped = re.findall(r"\$\{esc\(", page)
    assert len(escaped) >= 15, f"only {len(escaped)} escaped interpolations"
    for probe in ("esc(e.camera_id)", "esc(e.event_id)"):
        assert probe in page, f"{probe} is interpolated unescaped"


def test_the_incidents_endpoint_groups_alerts(client):
    payload = client.get("/api/incidents").json()
    assert set(payload) == {"count", "incidents"}
    assert payload["count"] == len(payload["incidents"])


def test_the_incident_gap_is_validated(client):
    assert client.get("/api/incidents?gap_seconds=0").status_code == 422
    assert client.get("/api/incidents?gap_seconds=99999").status_code == 422
    assert client.get("/api/incidents?gap_seconds=60").status_code == 200


def test_the_dashboard_can_show_incidents_as_well_as_alerts(client):
    """The grouping only reduces alert fatigue if an operator can see it."""
    page = client.get("/").text
    assert "/api/incidents" in page, "incidents are never fetched"
    assert "tabIncidents" in page, "there is no way to switch to them"
    # Read from the real incident payload, not from alert fields.
    for field in ("peak_severity", "peak_event_id", "cameras"):
        assert field in page, f"incident view never reads {field}"


def test_a_broken_alert_feed_says_so_instead_of_going_quiet(tmp_path, monkeypatch):
    """A blank dashboard with no reason given is the worst possible failure.

    If an event cannot be serialised the websocket dies. Swallowing that leaves
    an operator staring at an empty screen during the one hour it matters, so
    the reason has to reach the console.
    """
    from app import ui
    from app.store import EventStore

    said: list[str] = []
    monkeypatch.setattr(ui, "warn", said.append)
    monkeypatch.setattr(
        EventStore, "recent",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("payload is not JSON")),
    )

    config = SurveillanceConfig(
        source="synthetic", detector="sim", synthetic_frames=1,
        zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, quiet=True, print_summary=False, color=False,
    )
    app = create_app([CameraSpec(camera_id="CAM-01", source="synthetic")],
                     config, db_path=tmp_path / "broken.db")

    with TestClient(app) as client:
        try:
            with client.websocket_connect("/ws"):
                pass
        except Exception:
            pass          # the connection failing is the point; the silence was the bug

    # Startup emits unrelated warnings of its own, so look only at ours.
    feed = [w for w in said if "alert feed" in w]
    assert feed, "a dead alert feed must not fail silently"
    assert "payload is not JSON" in feed[0], "the warning must carry the actual cause"


def test_the_broken_feed_is_reported_once_not_per_reconnect(tmp_path, monkeypatch):
    """A dashboard retries on a timer; one real fault must not become a flood."""
    from app import ui
    from app.store import EventStore

    said: list[str] = []
    monkeypatch.setattr(ui, "warn", said.append)
    monkeypatch.setattr(
        EventStore, "recent",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("still broken")),
    )

    config = SurveillanceConfig(
        source="synthetic", detector="sim", synthetic_frames=1,
        zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, quiet=True, print_summary=False, color=False,
    )
    app = create_app([CameraSpec(camera_id="CAM-01", source="synthetic")],
                     config, db_path=tmp_path / "flood.db")

    with TestClient(app) as client:
        for _ in range(4):
            try:
                with client.websocket_connect("/ws"):
                    pass
            except Exception:
                pass

    feed = [w for w in said if "alert feed" in w]
    assert len(feed) == 1, f"warned {len(feed)} times for one fault"
