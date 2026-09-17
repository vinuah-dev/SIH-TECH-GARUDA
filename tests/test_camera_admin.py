"""Adding and removing cameras from the dashboard.

A camera used to join the fleet by editing JSON and restarting the server.
Now it joins from the website - a webcam, an IP camera, a wired camera on a
DVR, a phone's stream - and starts at once.

What these tests hold it to:

**Only camera sources, never files.** This is the one endpoint on an API with
no authentication that sets where video comes from. A source is a webcam
number, an rtsp/http camera address or the demo feed; a path would let anybody
who reaches the port make the server open any file on the machine.

**A password goes into the fleet file and nowhere else.** The camera needs it
to stream, so the file keeps it. Status, errors and the console never show it.

**Other websites cannot drive it.** A page on another site must not be able to
make an operator's browser add a camera.

**Nothing lands on disk the server could not load back**, the same staged
write the map editor uses.
"""

import json
import socket
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import camera_sources
from app.camera_sources import (ProbeFailed, describe_source, find_webcams,
                                grab_frame, normalize_source, redact_source)
from app.config import SurveillanceConfig
from app.runner import CameraSpec, MultiCameraRunner, load_camera_specs, load_site_map
from app.server import create_app

FLEET = {
    "_comment": "kept verbatim across saves",
    "cameras": [
        {"camera_id": "CAM-01", "source": "synthetic",
         "location": {"lat": 28.613939, "lon": 77.209023, "site": "Gate A"}},
    ],
    "geofences": [{
        "name": "SECTOR-7", "kind": "SECTOR",
        "polygon": [[28.61, 77.205], [28.618, 77.205], [28.618, 77.213], [28.61, 77.213]],
    }],
}


def config_for(tmp_path, frames=100000):
    return SurveillanceConfig(
        source="synthetic", detector="sim", synthetic_frames=frames,
        zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, quiet=True, print_summary=False, color=False,
    )


def closed_port() -> int:
    """A local port with nothing listening, so a connect is refused at once."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(autouse=True)
def no_real_webcam(monkeypatch):
    """Nothing in these tests may switch on the webcam of the machine running them."""
    import cv2

    real = cv2.VideoCapture

    def guarded(source, *args, **kwargs):
        if isinstance(source, int):
            raise AssertionError(f"a test tried to open real webcam {source}")
        return real(source, *args, **kwargs)

    monkeypatch.setattr(cv2, "VideoCapture", guarded)


@pytest.fixture
def fleet(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(FLEET, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def client(tmp_path, fleet):
    app = create_app(load_camera_specs(fleet), config_for(tmp_path),
                     db_path=tmp_path / "a.db", fleet_path=fleet,
                     geofences=load_site_map(fleet).geofences)
    with TestClient(app) as ready:
        ready.fleet = fleet
        yield ready


def on_disk(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw, {c["camera_id"]: c for c in raw["cameras"]}


def status_ids(client):
    status = client.get("/api/status").json()
    return ({c["camera_id"] for c in status["cameras"]},
            {s["camera_id"] for s in status["skipped"]})


# ----------------------------------------------------------------- sources


@pytest.mark.parametrize("given, stored", [
    ("0", "webcam:0"),
    ("webcam:1", "webcam:1"),
    ("WEBCAM:2", "webcam:2"),
    ("synthetic", "synthetic"),
    ("rtsp://admin:p%40ss@192.168.1.64:554/Streaming/Channels/101",
     "rtsp://admin:p%40ss@192.168.1.64:554/Streaming/Channels/101"),
    ("http://100.66.98.0:8080/video", "http://100.66.98.0:8080/video"),
    ("  https://cam.example/live  ", "https://cam.example/live"),
])
def test_camera_sources_the_dashboard_accepts(given, stored):
    assert normalize_source(given) == stored


@pytest.mark.parametrize("bad", [
    "", "   ", None,
    "C:/Windows/win.ini", "../data/sentinelx.db", "clip.mp4", "/etc/passwd",
    "file:///etc/passwd", "ftp://192.168.1.5/cam", "javascript:alert(1)",
    "rtsp://", "http:///video", "rtsp://host:99999/x",
    "webcam:10", "42",
    "rtsp://192.168.1.64/a b", "rtsp://192.168.1.64/x\ny",
    "rtsp://192.168.1.64/" + "x" * 500,
])
def test_anything_else_is_refused(bad):
    with pytest.raises(ValueError):
        normalize_source(bad)


def test_a_password_never_shows_in_what_is_displayed():
    shown = redact_source("rtsp://admin:hunter2@192.168.1.64:554/Streaming/Channels/101")
    assert "hunter2" not in shown
    assert shown == "rtsp://admin:***@192.168.1.64:554/Streaming/Channels/101"
    query = redact_source("http://10.0.0.9/videostream.cgi?user=admin&pwd=hunter2&rate=0")
    assert "hunter2" not in query and "user=admin" in query and "rate=0" in query
    assert redact_source("webcam:0") == "webcam:0"
    assert redact_source("http://100.66.98.0:8080/video") == "http://100.66.98.0:8080/video"


def test_sources_get_a_label_a_person_can_read():
    assert describe_source("webcam:1") == "Webcam 1"
    assert describe_source("synthetic") == "Demo feed"
    assert describe_source("rtsp://a:b@192.168.1.64:554/x") == "RTSP · 192.168.1.64:554"
    assert describe_source("data/demo/gate.mp4") == "Video file · gate.mp4"


# ----------------------------------------------------------------- probing


def test_testing_the_demo_feed_returns_a_picture(client):
    body = client.post("/api/cameras/test", json={"source": "synthetic"}).json()
    assert body["ok"] is True
    assert body["width"] > 0 and body["height"] > 0
    assert body["preview"].startswith("data:image/jpeg;base64,")


def test_a_camera_that_does_not_answer_says_so_without_the_password(client):
    source = f"rtsp://admin:hunter2@127.0.0.1:{closed_port()}/Streaming/Channels/101"
    response = client.post("/api/cameras/test", json={"source": source})
    assert response.status_code == 200, "a typo is a normal answer, not a server error"
    body = response.json()
    assert body["ok"] is False
    assert "nothing answered" in body["reason"]
    assert "hunter2" not in response.text


def test_a_login_the_camera_refuses_is_told_apart_from_no_answer(monkeypatch):
    import app.video.stream as stream

    class Refused:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(stream, "tcp_reachable", lambda url, timeout: True)
    monkeypatch.setattr(stream, "open_network_capture", lambda url: Refused())
    with pytest.raises(ProbeFailed, match="username, password"):
        grab_frame("rtsp://admin:wrong@192.168.1.64:554/x")


def test_testing_a_webcam_a_camera_already_uses_does_not_open_it(tmp_path, fleet):
    raw = json.loads(fleet.read_text(encoding="utf-8"))
    raw["cameras"].append({"camera_id": "DESK", "source": "synthetic"})
    fleet.write_text(json.dumps(raw), encoding="utf-8")
    specs = load_camera_specs(fleet)
    specs[1].source = "webcam:0"          # in memory only: nothing opens a real webcam
    app = create_app(specs, config_for(tmp_path, frames=5), db_path=tmp_path / "w.db",
                     fleet_path=fleet)
    app.state.service.start = lambda: None   # never start it: that would open the webcam
    with TestClient(app) as c:
        body = c.post("/api/cameras/test", json={"source": "webcam:0"}).json()
    assert body["ok"] is False and "in use by DESK" in body["reason"]


def test_finding_webcams_never_opens_one_that_is_in_use():
    opened = []

    class Fake:
        def __init__(self, index):
            opened.append(index)
            self.index = index

        def isOpened(self):
            return self.index == 1

        def read(self):
            return True, np.zeros((480, 640, 3), np.uint8)

        def release(self):
            pass

    found = find_webcams({0: "CAM-01"}, indices=range(3), opener=Fake)
    assert 0 not in opened, "a webcam a camera holds must not be opened again"
    assert found == [{"index": 0, "in_use_by": "CAM-01"},
                     {"index": 1, "width": 640, "height": 480}]


# ------------------------------------------------------------------ adding


def test_a_camera_added_from_the_dashboard_starts_and_is_saved(client):
    response = client.post("/api/cameras", json={"camera_id": "CAM-02", "source": "synthetic"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["running"] is True and body["saved"] is True
    assert body["label"] == "Demo feed"

    raw, cameras = on_disk(client.fleet)
    assert cameras["CAM-02"] == {"camera_id": "CAM-02", "source": "synthetic"}
    assert raw["_comment"] == FLEET["_comment"]
    assert raw["geofences"] == FLEET["geofences"]
    assert cameras["CAM-01"]["location"]["site"] == "Gate A"

    running, _ = status_ids(client)
    assert running == {"CAM-01", "CAM-02"}, "no restart needed"
    assert "CAM-02" in client.get("/api/map").json()["unplaced"]
    assert client.get("/api/cameras/CAM-02/zones").status_code == 200
    # And it is what the server loads next time.
    assert [s.camera_id for s in load_camera_specs(client.fleet)] == ["CAM-01", "CAM-02"]


def test_an_added_camera_shares_the_fleet(client):
    client.post("/api/cameras", json={"camera_id": "CAM-02", "source": "synthetic"})
    runner = client.app.state.service.runner
    added = next(w for w in runner.workers if w.spec.camera_id == "CAM-02")
    # One person ledger, so a subject can be followed onto the new camera too.
    assert added.pipeline.people is runner.person_ledger


def test_a_camera_that_is_down_is_saved_and_shown_offline(client):
    source = f"rtsp://admin:hunter2@127.0.0.1:{closed_port()}/cam/realmonitor?channel=2&subtype=1"
    response = client.post("/api/cameras", json={"camera_id": "DVR-CH2", "source": source})
    assert response.status_code == 200
    body = response.json()
    assert body["running"] is False and "not answering" in body["reason"]
    assert "hunter2" not in response.text

    _, cameras = on_disk(client.fleet)
    assert cameras["DVR-CH2"]["source"] == source, "the camera needs its login to stream"

    status = client.get("/api/status")
    assert "hunter2" not in status.text
    skipped = {s["camera_id"]: s for s in status.json()["skipped"]}
    assert skipped["DVR-CH2"]["source"].startswith("rtsp://admin:***@127.0.0.1")
    assert skipped["DVR-CH2"]["label"].startswith("RTSP · 127.0.0.1")


@pytest.mark.parametrize("body, code", [
    ({"camera_id": "CAM-01", "source": "synthetic"}, 409),        # name taken
    ({"camera_id": "", "source": "synthetic"}, 400),
    ({"camera_id": "../../evil", "source": "synthetic"}, 400),
    ({"camera_id": "CAM 02", "source": "synthetic"}, 400),
    ({"camera_id": "CAM-02", "source": "config/cameras.json"}, 400),
    ({"camera_id": "CAM-02", "source": "file:///C:/Windows/win.ini"}, 400),
    ({"camera_id": "CAM-02", "source": ""}, 400),
    ({"camera_id": "CAM-02"}, 400),
])
def test_a_bad_camera_is_refused_and_nothing_is_written(client, body, code):
    before = client.fleet.read_text(encoding="utf-8")
    assert client.post("/api/cameras", json=body).status_code == code
    assert client.fleet.read_text(encoding="utf-8") == before
    assert not list(client.fleet.parent.glob("*.saving"))


def test_only_a_name_and_a_source_are_taken_from_the_request(client):
    client.post("/api/cameras", json={
        "camera_id": "CAM-02", "source": "synthetic",
        "zones": "../../somewhere.json", "detector": "yolo", "events_dir": "C:/",
    })
    _, cameras = on_disk(client.fleet)
    assert cameras["CAM-02"] == {"camera_id": "CAM-02", "source": "synthetic"}


def test_two_cameras_cannot_share_one_webcam(tmp_path, fleet):
    specs = load_camera_specs(fleet)
    specs[0].source = "webcam:0"          # in memory only: nothing opens a real webcam
    app = create_app(specs, config_for(tmp_path, frames=5), db_path=tmp_path / "s.db",
                     fleet_path=fleet)
    app.state.service.start = lambda: None   # never start it: that would open the webcam
    with TestClient(app) as c:
        response = c.post("/api/cameras", json={"camera_id": "CAM-02", "source": "0"})
    assert response.status_code == 409 and "already used by CAM-01" in response.json()["detail"]


def test_another_website_cannot_add_a_camera(client):
    before = client.fleet.read_text(encoding="utf-8")
    response = client.post("/api/cameras", json={"camera_id": "CAM-02", "source": "synthetic"},
                           headers={"Origin": "http://evil.example"})
    assert response.status_code == 403
    response = client.post("/api/cameras", json={"camera_id": "CAM-02", "source": "synthetic"},
                           headers={"Sec-Fetch-Site": "cross-site"})
    assert response.status_code == 403
    # A form or no-cors fetch cannot send JSON; one without a content type is refused too.
    response = client.post("/api/cameras",
                           content=json.dumps({"camera_id": "CAM-02", "source": "synthetic"}))
    assert response.status_code in (415, 422)
    assert client.fleet.read_text(encoding="utf-8") == before
    assert client.post("/api/cameras/test", json={"source": "synthetic"},
                       headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.delete("/api/cameras/CAM-01",
                         headers={"Origin": "http://evil.example"}).status_code == 403


def test_the_dashboard_itself_is_let_through(client):
    host = "testserver"
    response = client.post("/api/cameras", json={"camera_id": "CAM-02", "source": "synthetic"},
                           headers={"Origin": f"http://{host}", "Sec-Fetch-Site": "same-origin"})
    assert response.status_code == 200, response.text


def test_without_a_fleet_file_cameras_cannot_be_added(tmp_path, fleet):
    app = create_app(load_camera_specs(fleet), config_for(tmp_path, frames=5),
                     db_path=tmp_path / "n.db")
    with TestClient(app) as bare:
        assert bare.get("/api/status").json()["fleet_editable"] is False
        response = bare.post("/api/cameras", json={"camera_id": "CAM-02", "source": "synthetic"})
        assert response.status_code == 409
        assert "--cameras" in response.json()["detail"]


# ---------------------------------------------------------------- removing


def test_a_camera_removed_from_the_dashboard_stops_and_leaves_the_file(client):
    client.post("/api/cameras", json={"camera_id": "CAM-02", "source": "synthetic"})
    client.put("/api/map/cameras/CAM-02", json={"lat": 28.6146, "lon": 77.2099})

    response = client.delete("/api/cameras/CAM-02")
    assert response.status_code == 200, response.text
    assert response.json()["was_running"] is True

    raw, cameras = on_disk(client.fleet)
    assert set(cameras) == {"CAM-01"}
    assert raw["geofences"] == FLEET["geofences"]
    running, skipped = status_ids(client)
    assert "CAM-02" not in running | skipped
    assert "CAM-02" not in {c["camera_id"] for c in client.get("/api/map").json()["cameras"]}
    assert client.get("/api/cameras/CAM-02/snapshot").status_code == 404


def test_an_offline_camera_can_be_removed_too(client):
    source = f"rtsp://127.0.0.1:{closed_port()}/stream1"
    client.post("/api/cameras", json={"camera_id": "TAPO", "source": source})
    assert client.delete("/api/cameras/TAPO").status_code == 200
    _, skipped = status_ids(client)
    assert "TAPO" not in skipped


def test_the_last_camera_cannot_be_removed(client):
    """A fleet file with no cameras is one the server refuses to start with."""
    response = client.delete("/api/cameras/CAM-01")
    assert response.status_code == 409
    assert "at least one camera" in response.json()["detail"]
    assert "CAM-01" in on_disk(client.fleet)[1]


def test_removing_a_camera_that_does_not_exist_is_a_404(client):
    assert client.delete("/api/cameras/NOPE").status_code == 404


# ------------------------------------------------------------------ runner


def test_a_network_camera_that_is_down_still_stops_when_asked():
    """It never delivers the frame that would let its pipeline see a stop flag."""
    from app.video.stream import StreamSource

    source = StreamSource(url="rtsp://10.255.255.1/x", probe=lambda url, timeout: False,
                          sleep=lambda seconds: time.sleep(0.01))
    reader = threading.Thread(target=lambda: list(source.frames()), daemon=True)
    reader.start()
    time.sleep(0.1)
    assert reader.is_alive(), "it should be retrying"
    source.stop()
    reader.join(timeout=2)
    assert not reader.is_alive()


def test_the_runner_adds_and_removes_cameras_while_running(tmp_path):
    runner = MultiCameraRunner([CameraSpec(camera_id="CAM-01", source="synthetic")],
                               config_for(tmp_path), capture_frames=True)
    runner.start()
    try:
        assert runner.add_camera(CameraSpec(camera_id="CAM-02", source="synthetic")) == (True, "")
        assert [w.spec.camera_id for w in runner.workers] == ["CAM-01", "CAM-02"]
        with pytest.raises(ValueError):
            runner.add_camera(CameraSpec(camera_id="CAM-02", source="synthetic"))

        deadline = time.time() + 10
        while runner.latest_frame("CAM-02") is None and time.time() < deadline:
            time.sleep(0.05)
        assert runner.latest_frame("CAM-02") is not None, "the new camera produces frames"

        assert runner.remove_camera("CAM-02") is True
        assert [w.spec.camera_id for w in runner.workers] == ["CAM-01"]
        assert runner.latest_frame("CAM-02") is None
    finally:
        runner.stop()


def test_status_never_carries_a_password(tmp_path):
    spec = CameraSpec(camera_id="CAM-01", source="rtsp://admin:hunter2@10.0.0.9/x")
    runner = MultiCameraRunner([spec], config_for(tmp_path))
    from app.runner import CameraWorker
    from app.pipeline import SurveillancePipeline

    worker = CameraWorker(spec=spec, pipeline=SurveillancePipeline(spec.build_config(runner.base_config)))
    status = worker.status()
    assert "hunter2" not in json.dumps(status)
    assert status["label"] == "RTSP · 10.0.0.9"


# -------------------------------------------------------------------- page


def test_the_dashboard_has_the_add_camera_window():
    from pathlib import Path

    page = Path("app/server/static/index.html").read_text(encoding="utf-8")
    for hook in ('id="addScrim"', 'data-kind="webcam"', 'data-kind="ip"', 'data-kind="dvr"',
                 'data-kind="url"', 'data-kind="demo"', "/api/cameras/test",
                 "/api/cameras/webcams", '"POST", "/api/cameras"', "data-remove",
                 "fleet_editable"):
        assert hook in page, f"the add-camera window lost {hook}"
    # Brand paths people would otherwise have to look up.
    assert "/Streaming/Channels/" in page and "/cam/realmonitor?channel=" in page
