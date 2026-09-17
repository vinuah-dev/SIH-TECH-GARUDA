"""Streaming a camera's annotated view to the dashboard.

The dashboard shows the same picture the desktop window shows - boxes, zones,
track ids and the live risk read-out - so an operator is looking at what the
system is actually deciding on, not a raw feed beside it.

Two things here are easy to get wrong and silent when wrong. A browser shown
malformed MJPEG framing displays nothing at all, with no error anywhere. And a
pipeline that only draws when a window is open serves a blank dashboard, which
looks exactly like a camera that is down.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import SurveillanceConfig
from app.runner import CameraSpec, MultiCameraRunner
from app.server import create_app

JPEG_MAGIC = bytes((0xFF, 0xD8, 0xFF))


def config(tmp_path, **kwargs):
    settings = dict(
        source="synthetic", detector="sim", synthetic_frames=40, force_night=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, quiet=True, print_summary=False, color=False,
    )
    settings.update(kwargs)
    return SurveillanceConfig(**settings)


# --------------------------------------------------------------- the framing


def test_the_multipart_framing_is_exactly_what_a_browser_expects():
    """Get a CRLF wrong and the browser shows nothing, with no error to say why."""
    import app.server.api as api

    source = (api.__file__ and open(api.__file__, encoding="utf-8").read())
    assert "CRLF = bytes((13, 10))" in source, (
        "the framing must be built from explicit byte values - escape sequences "
        "through a shell have silently become real newlines here before"
    )


def test_the_stream_route_is_registered(tmp_path):
    """Checked on the route table rather than by opening it.

    The endpoint is an endless generator by design - a live feed has no last
    frame - so a test client that reads it to completion never returns. The
    stream is exercised for real against a running server instead; what a unit
    test can honestly assert is that the route exists and is shaped right.
    """
    app = create_app([CameraSpec(camera_id="CAM-01", source="synthetic")],
                     config(tmp_path), db_path=tmp_path / "s.db")
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/api/cameras/{camera_id}/stream" in paths


def test_an_unknown_camera_is_a_404_not_an_empty_stream(tmp_path):
    """A blank tile and a wrong camera id must not look the same."""
    app = create_app([CameraSpec(camera_id="CAM-01", source="synthetic")],
                     config(tmp_path), db_path=tmp_path / "s.db")
    with TestClient(app) as client:
        response = client.get("/api/cameras/NOPE/stream")
        assert response.status_code == 404
        assert "NOPE" in response.json()["detail"]


# -------------------------------------------------------- frames being kept


def test_the_fleet_keeps_frames_for_the_dashboard_without_a_window():
    """A pipeline that only draws when a window is open serves a blank dashboard."""
    runner = MultiCameraRunner(
        [CameraSpec(camera_id="CAM-01", source="synthetic")],
        SurveillanceConfig(source="synthetic", detector="sim", view=False,
                           zones_path="config/zones.json", color=False),
        capture_frames=True,
    )
    runner.detector = None
    worker = runner._build_workers()[0]
    assert worker.pipeline.frame_sink is not None, (
        "no sink means no frames reach the dashboard"
    )


def test_without_capture_and_without_a_window_no_frames_are_kept():
    """Keeping frames costs memory and a draw per frame; it is not free."""
    runner = MultiCameraRunner(
        [CameraSpec(camera_id="CAM-01", source="synthetic")],
        SurveillanceConfig(source="synthetic", detector="sim", view=False,
                           zones_path="config/zones.json", color=False),
    )
    runner.detector = None
    assert runner._build_workers()[0].pipeline.frame_sink is None


def test_the_newest_frame_is_the_one_served():
    """A viewer falling behind should see live video, not catch up on a backlog."""
    runner = MultiCameraRunner(
        [CameraSpec(camera_id="CAM-01", source="synthetic")],
        SurveillanceConfig(source="synthetic", detector="sim", color=False),
        capture_frames=True,
    )
    first = np.zeros((8, 8, 3), np.uint8)
    second = np.full((8, 8, 3), 255, np.uint8)
    runner._post_frame("CAM-01", first)
    runner._post_frame("CAM-01", second)

    assert runner.latest_frame("CAM-01") is second
    assert runner.latest_frame("CAM-02") is None


def test_the_dashboard_asks_the_fleet_to_keep_frames(tmp_path):
    app = create_app([CameraSpec(camera_id="CAM-01", source="synthetic")],
                     config(tmp_path), db_path=tmp_path / "s.db")
    with TestClient(app):
        assert app.state.service.runner.capture_frames is True


# ------------------------------------------------------ what gets drawn on


def test_the_streamed_frame_is_the_annotated_one_not_the_raw_feed(tmp_path):
    """An operator must see what the system decided on, not a bare picture."""
    from app.pipeline import SurveillancePipeline

    drawn = []
    pipeline = SurveillancePipeline(
        config(tmp_path, synthetic_frames=25),
        frame_sink=lambda camera_id, frame: drawn.append((camera_id, frame)),
    )
    pipeline.run()

    assert drawn, "nothing was posted to the sink"
    camera_id, frame = drawn[-1]
    assert camera_id == "CAM-01"
    assert frame.ndim == 3, "an annotated frame is a colour image"
    # The zone overlay and HUD are drawn in colour on a greyscale-ish scene, so
    # a raw frame and an annotated one cannot have identical channels.
    assert not (frame[..., 0] == frame[..., 2]).all(), (
        "this looks like an unannotated frame"
    )


def test_frames_are_posted_every_frame_not_only_when_something_alarms(tmp_path):
    """A view that only updates on an alert is a slideshow, not a live feed."""
    from app.pipeline import SurveillancePipeline

    drawn = []
    stats = SurveillancePipeline(
        config(tmp_path, synthetic_frames=30),
        frame_sink=lambda camera_id, frame: drawn.append(frame),
    ).run()

    assert stats.alerts <= 2, "this fixture should not be all alerts"
    assert len(drawn) >= stats.frames - 1, (
        f"{len(drawn)} frames posted from {stats.frames} - the sink is being skipped"
    )


# ------------------------------------------------- a camera that says nothing


def test_a_camera_that_delivered_no_frames_explains_itself():
    """STOPPED with no reason, next to a camera that gives one, is the worst case.

    A webcam already held by another program opens fine and then delivers
    nothing. That used to end silently: no exception, no error field, just a
    dead tile. It is also the single most likely failure in a live demo, and
    the fix for it - close the other program - is nothing like the fix for a
    camera that is off the network.
    """
    from app.pipeline import SessionStats
    from app.runner import CameraWorker

    worker = CameraWorker.__new__(CameraWorker)
    worker.spec = CameraSpec(camera_id="CAM-01", source="webcam:0")
    worker.error = None
    worker.thread = None

    reason = worker._explain(SessionStats())
    assert reason, "a silent stop must still say something"
    assert "webcam:0" in reason, "the reason should name the source"
    assert "another program" in reason


def test_a_camera_that_ran_is_not_reported_as_broken():
    from app.pipeline import SessionStats
    from app.runner import CameraWorker

    worker = CameraWorker.__new__(CameraWorker)
    worker.spec = CameraSpec(camera_id="CAM-01", source="webcam:0")
    worker.error = None
    worker.thread = None

    stats = SessionStats()
    stats.frames = 120
    assert worker._explain(stats) is None


def test_a_real_exception_is_reported_verbatim():
    """A raised error already says more than any guess could."""
    from app.pipeline import SessionStats
    from app.runner import CameraWorker

    worker = CameraWorker.__new__(CameraWorker)
    worker.spec = CameraSpec(camera_id="CAM-01", source="webcam:0")
    worker.error = RuntimeError("codec not supported")
    worker.thread = None

    assert worker._explain(SessionStats()) == "codec not supported"


# ---------------------------------------------------------- browser budget


def test_the_dashboard_never_streams_every_camera_at_once():
    """A browser gives one server six connections, and an MJPEG stream keeps one.

    With a stream per tile, six cameras used them all up: alert polling, the
    map and the zone editor then waited forever, with no error anywhere. The
    tiles must not open their streams by themselves, and a fixed number may.
    """
    import re
    from pathlib import Path

    page = Path("app/server/static/index.html").read_text(encoding="utf-8")
    assert not re.search(r'<img[^>]+src="/api/cameras/\$\{[^}]+\}/stream"', page), (
        "a camera tile opens its own stream again - six cameras will freeze the dashboard"
    )
    budget = re.search(r"const LIVE_STREAMS = (\d+);", page)
    assert budget, "the number of live streams must be capped"
    assert 1 <= int(budget.group(1)) <= 3, (
        "polling alone fires four requests at once; more than two streams starves it"
    )


def test_streams_close_when_the_wall_is_not_on_screen():
    """The map and zone pages need connections the wall would otherwise hold."""
    from pathlib import Path

    page = Path("app/server/static/index.html").read_text(encoding="utf-8")
    assert 'addEventListener("visibilitychange", syncStreams)' in page
    body = page[page.index("function showView"):page.index("document.querySelectorAll(\".nav a[data-view]\")")]
    assert "syncStreams()" in body, "changing page must close or reopen the streams"
