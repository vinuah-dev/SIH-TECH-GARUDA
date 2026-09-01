"""Live stream ingestion: reconnect behaviour, backoff and health reporting.

A fake capture stands in for cv2.VideoCapture so the whole drop/reconnect
state machine is testable without a camera or a network.

Note the use of `islice` below. A live stream has no end - a failed read is a
dropped link, not EOF - so an endless source is correct behaviour and the
*consumer* is what stops it, exactly as the pipeline does with --limit or
Ctrl+C. Tests that want a finite stream either take N frames or configure a
finite `reconnect_attempts`.
"""

from itertools import islice

import numpy as np
import pytest

from app.video.sources import open_source
from app.video.stream import CameraState, StreamSource, is_network_source

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeCapture:
    """Yields `script` entries in order: True = a frame, False = a read failure."""

    def __init__(self, script, opens=True):
        self.script = list(script)
        self._opens = opens
        self.released = False
        self.props: dict[int, float] = {}

    def isOpened(self):
        return self._opens

    def read(self):
        if not self.script:
            return False, None
        ok = self.script.pop(0)
        return (True, FRAME.copy()) if ok else (False, None)

    def release(self):
        self.released = True

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return 0.0


def factory(*captures):
    """Hands out one capture per connection attempt, in order."""
    queue = list(captures)

    def make(url):
        return queue.pop(0) if queue else FakeCapture([], opens=False)

    return make


def make_source(capture_factory, **kwargs):
    slept: list[float] = []
    source = StreamSource(
        url="rtsp://camera/stream",
        capture_factory=capture_factory,
        probe=lambda url, timeout: True,
        sleep=slept.append,
        reconnect_backoff=1.0,
        **kwargs,
    )
    return source, slept


# ------------------------------------------------------------------ routing


@pytest.mark.parametrize("url", ["rtsp://cam/1", "rtsps://cam/1", "http://cam", "RTSP://CAM"])
def test_network_urls_are_recognised(url):
    assert is_network_source(url)


@pytest.mark.parametrize("spec", ["webcam:0", "synthetic", "clip.mp4", "0"])
def test_local_sources_are_not_network(spec):
    assert not is_network_source(spec)


def test_open_source_routes_rtsp_to_a_stream():
    source = open_source("rtsp://camera/stream")
    assert isinstance(source, StreamSource)
    assert source.kind == "RTSP"


# ------------------------------------------------------------- happy stream


def test_reads_frames_from_a_healthy_stream():
    source, _ = make_source(factory(FakeCapture([True] * 10)))
    frames = list(islice(source.frames(), 3))
    assert len(frames) == 3
    assert source.health.state is CameraState.ONLINE
    assert source.health.frames_read == 3
    assert source.health.connects == 1


def test_limit_stops_the_stream():
    source, _ = make_source(factory(FakeCapture([True] * 10)), limit=4)
    assert len(list(source.frames())) == 4


def test_stride_thins_the_stream():
    source, _ = make_source(factory(FakeCapture([True] * 20)), stride=3, limit=3)
    assert len(list(source.frames())) == 3


def test_stream_time_starts_at_zero_and_advances():
    ticks = iter([100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
    source = StreamSource(
        url="rtsp://camera/stream",
        capture_factory=factory(FakeCapture([True, True])),
        probe=lambda url, timeout: True,
        sleep=lambda _: None,
        clock=lambda: next(ticks),
    )
    times = [f.stream_time for f in islice(source.frames(), 2)]
    assert times[0] >= 0
    assert times[1] > times[0]


def test_buffer_is_kept_at_one_frame():
    """Live analysis must work on the newest frame, not a backlog."""
    import cv2

    capture = FakeCapture([True] * 5)
    source, _ = make_source(factory(capture))
    list(islice(source.frames(), 1))
    assert capture.props.get(cv2.CAP_PROP_BUFFERSIZE) == 1


# --------------------------------------------------------------- reconnect


def test_a_dropped_stream_reconnects_and_keeps_going():
    first = FakeCapture([True, False])       # one frame, then the link drops
    second = FakeCapture([True] * 10)        # reconnected
    source, _ = make_source(factory(first, second))

    frames = list(islice(source.frames(), 3))
    assert len(frames) == 3
    assert source.health.disconnects == 1
    assert source.health.connects == 2
    assert first.released


def test_frame_indices_stay_continuous_across_a_reconnect():
    source, _ = make_source(factory(FakeCapture([True, False]), FakeCapture([True] * 5)))
    assert [f.index for f in islice(source.frames(), 3)] == [0, 1, 2]


def test_a_camera_that_never_opens_is_retried_then_given_up_on():
    dead = [FakeCapture([], opens=False) for _ in range(6)]
    source, slept = make_source(factory(*dead), reconnect_attempts=3)

    assert list(source.frames()) == []
    assert source.health.state is CameraState.OFFLINE
    assert "gave up after 3" in source.health.last_error
    assert len(slept) == 3


def test_backoff_grows_exponentially_and_is_capped():
    dead = [FakeCapture([], opens=False) for _ in range(8)]
    source, slept = make_source(
        factory(*dead), reconnect_attempts=6, max_backoff=4.0
    )
    list(source.frames())
    assert slept == [1.0, 2.0, 4.0, 4.0, 4.0, 4.0]


def test_backoff_resets_after_a_successful_reconnect():
    source, slept = make_source(
        factory(
            FakeCapture([True, False]),
            FakeCapture([], opens=False),
            FakeCapture([True, False]),
            FakeCapture([True] * 5),
        )
    )
    list(islice(source.frames(), 3))
    # Each reconnect sequence starts again at the base delay.
    assert slept[0] == 1.0
    assert 1.0 in slept[1:]


def test_release_stops_a_reconnecting_stream():
    source, _ = make_source(factory(FakeCapture([], opens=False)))
    source.release()
    assert list(source.frames()) == []
    assert source.health.state is CameraState.OFFLINE


def test_an_exception_while_opening_is_recorded_not_raised():
    def explode(url):
        raise OSError("connection refused")

    source, _ = make_source(explode, reconnect_attempts=1)
    assert list(source.frames()) == []
    assert source.health.last_error is not None


def test_an_exception_while_reading_triggers_a_reconnect():
    class Exploding(FakeCapture):
        def read(self):
            raise OSError("stream reset")

    source, _ = make_source(factory(Exploding([]), FakeCapture([True] * 5)))
    assert len(list(islice(source.frames(), 1))) == 1
    assert source.health.disconnects == 1


# ------------------------------------------------------------------ health


def test_state_changes_are_reported():
    seen: list[CameraState] = []
    source = StreamSource(
        url="rtsp://camera/stream",
        capture_factory=factory(FakeCapture([True, False]), FakeCapture([True] * 5)),
        probe=lambda url, timeout: True,
        sleep=lambda _: None,
        on_state_change=lambda health: seen.append(health.state),
    )
    list(islice(source.frames(), 2))
    assert CameraState.CONNECTING in seen
    assert CameraState.ONLINE in seen
    assert CameraState.RECONNECTING in seen


def test_health_serializes_for_the_api():
    source, _ = make_source(factory(FakeCapture([True] * 5)))
    list(islice(source.frames(), 1))
    payload = source.health.to_dict()
    assert payload["state"] == "ONLINE"
    assert payload["frames_read"] == 1
    assert set(payload) == {
        "state", "connects", "disconnects", "frames_read", "last_frame_at", "last_error",
    }


# --------------------------------------------------- reachability pre-check


def test_an_unreachable_camera_fails_without_waiting_for_ffmpeg():
    """The pre-check is what stops a dead host blocking for the OS SYN timeout."""
    opened: list[str] = []

    def factory_that_should_not_run(url):
        opened.append(url)
        return FakeCapture([True])

    source = StreamSource(
        url="rtsp://10.255.255.1:554/stream",
        capture_factory=factory_that_should_not_run,
        probe=lambda url, timeout: False,
        sleep=lambda _: None,
        reconnect_attempts=1,
    )
    assert list(source.frames()) == []
    assert opened == []  # FFmpeg was never handed a dead address
    assert "10.255.255.1:554 unreachable" in source.health.last_error


def test_probe_uses_the_configured_open_timeout():
    seen: list[float] = []

    def probe(url, timeout):
        seen.append(timeout)
        return False

    source = StreamSource(
        url="rtsp://cam/1",
        probe=probe,
        sleep=lambda _: None,
        reconnect_attempts=1,
        open_timeout=3.0,
    )
    list(source.frames())
    assert seen and all(t == 3.0 for t in seen)


@pytest.mark.parametrize(
    "url,port",
    [
        ("rtsp://cam/1", 554),
        ("rtsp://cam:8554/1", 8554),
        ("http://cam/feed", 80),
        ("https://cam/feed", 443),
    ],
)
def test_default_ports_per_scheme(url, port):
    from urllib.parse import urlparse

    from app.video.stream import DEFAULT_PORTS

    parsed = urlparse(url)
    assert (parsed.port or DEFAULT_PORTS[parsed.scheme]) == port


def test_a_url_without_a_host_skips_the_probe():
    from app.video.stream import tcp_reachable

    assert tcp_reachable("rtsp:///no-host", timeout=0.1) is True
