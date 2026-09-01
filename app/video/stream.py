"""Live network camera ingestion (RTSP / HTTP).

A file always reads to the end; a border camera does not. It drops out, the
network stalls, the NVR reboots. This module is the difference between a demo
that reads a video file and something that can sit on a post for a week:

  * fails fast instead of blocking forever on a dead host
  * forces RTSP over TCP, because UDP on a long rural link loses packets
  * keeps the capture buffer at one frame, so analysis works on *now* rather
    than replaying a backlog it built up while the model was busy
  * reconnects with exponential backoff, and reports what it is doing

The capture object is injected so the reconnect logic can be tested without a
camera - see `capture_factory`.
"""

from __future__ import annotations

import os
import socket
import time
from urllib.parse import urlparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator, Protocol

import cv2

from .sources import Frame, _resize

# FFmpeg options applied to every network capture. `timeout` is in
# microseconds and is what stops a dead camera from hanging the process
# forever; `stimeout` is the older spelling, kept for older FFmpeg builds.
FFMPEG_OPTIONS = "rtsp_transport;tcp|timeout;5000000|stimeout;5000000"

NETWORK_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://")

DEFAULT_PORTS = {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443}


class CameraState(str, Enum):
    CONNECTING = "CONNECTING"
    ONLINE = "ONLINE"
    RECONNECTING = "RECONNECTING"
    OFFLINE = "OFFLINE"


@dataclass
class CameraHealth:
    """What the operator needs to know about the link, not the footage."""

    state: CameraState = CameraState.OFFLINE
    connects: int = 0
    disconnects: int = 0
    frames_read: int = 0
    last_frame_at: float | None = None
    last_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "connects": self.connects,
            "disconnects": self.disconnects,
            "frames_read": self.frames_read,
            "last_frame_at": self.last_frame_at,
            "last_error": self.last_error,
        }


class Capture(Protocol):
    """The slice of cv2.VideoCapture this module actually uses."""

    def isOpened(self) -> bool: ...
    def read(self): ...
    def release(self) -> None: ...
    def set(self, prop: int, value: float) -> bool: ...
    def get(self, prop: int) -> float: ...


def is_network_source(spec: str) -> bool:
    return spec.lower().startswith(NETWORK_SCHEMES)


def tcp_reachable(url: str, timeout: float) -> bool:
    """Fast reachability pre-check before handing the URL to FFmpeg.

    FFmpeg's `timeout` option governs socket *reads*, not the initial TCP
    connect, so an unroutable camera address blocks for the OS SYN timeout -
    around 30 seconds on Windows - per attempt. A plain socket connect with our
    own timeout turns that into a prompt, clearly worded failure.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return True  # Nothing to probe; let FFmpeg decide.
    port = parsed.port or DEFAULT_PORTS.get(parsed.scheme.lower(), 554)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def open_network_capture(url: str) -> Capture:
    """Open a network stream with the options that make RTSP behave."""
    # OpenCV reads this env var when the FFmpeg backend starts, so it has to
    # be set before the VideoCapture is constructed.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", FFMPEG_OPTIONS)
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


@dataclass
class StreamSource:
    """A reconnecting live camera feed."""

    url: str
    kind: str = "RTSP"
    stride: int = 1
    limit: int | None = None
    max_width: int | None = 960
    reconnect_attempts: int = 0        # 0 means keep trying forever
    reconnect_backoff: float = 2.0
    max_backoff: float = 30.0
    open_timeout: float = 10.0
    capture_factory: Callable[[str], Capture] = open_network_capture
    # Returning True always disables the pre-check (used by tests).
    probe: Callable[[str, float], bool] = tcp_reachable
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.time
    on_state_change: Callable[[CameraHealth], None] | None = None

    health: CameraHealth = field(default_factory=CameraHealth, init=False)

    def __post_init__(self) -> None:
        self.name = self.url
        self.fps = 0.0
        self._capture: Capture | None = None
        self._stop = False

    # ------------------------------------------------------------ lifecycle

    def _set_state(self, state: CameraState, error: str | None = None) -> None:
        if error:
            self.health.last_error = error
        if self.health.state != state:
            self.health.state = state
            if self.on_state_change:
                self.on_state_change(self.health)

    def _connect(self) -> bool:
        """One connection attempt. True if the stream is usable."""
        if not self.probe(self.url, self.open_timeout):
            parsed = urlparse(self.url)
            target = f"{parsed.hostname}:{parsed.port or DEFAULT_PORTS.get(parsed.scheme, 554)}"
            self._set_state(
                self.health.state,
                error=f"{target} unreachable within {self.open_timeout:.0f}s",
            )
            return False

        capture = None
        try:
            capture = self.capture_factory(self.url)
            if capture.isOpened():
                self._capture = capture
                self.health.connects += 1
                try:
                    # Analyse the newest frame, never a stale backlog.
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:  # pragma: no cover - some backends refuse
                    pass
                try:
                    self.fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
                except Exception:
                    self.fps = 0.0
                self._set_state(CameraState.ONLINE)
                return True
            capture.release()
        except Exception as exc:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            self._set_state(self.health.state, error=str(exc))
        return False

    def _reconnect_loop(self) -> bool:
        """Retry with exponential backoff. False once attempts are exhausted."""
        delay = self.reconnect_backoff
        attempt = 0
        while not self._stop:
            attempt += 1
            if self.reconnect_attempts and attempt > self.reconnect_attempts:
                # Keep the underlying reason: "gave up" alone tells an operator
                # nothing about whether the camera is unreachable, refusing
                # auth, or serving a codec we cannot open.
                cause = self.health.last_error
                summary = f"gave up after {attempt - 1} reconnect attempts"
                self._set_state(
                    CameraState.OFFLINE,
                    error=f"{summary} (last error: {cause})" if cause else summary,
                )
                return False

            self._set_state(CameraState.RECONNECTING)
            self.sleep(delay)
            if self._connect():
                return True
            delay = min(self.max_backoff, delay * 2)
        return False

    # --------------------------------------------------------------- frames

    def frames(self) -> Iterator[Frame]:
        if self._stop:
            return
        started = self.clock()
        emitted = 0
        raw_index = 0

        self._set_state(CameraState.CONNECTING)
        if not self._connect() and not self._reconnect_loop():
            return

        while not self._stop:
            ok, image = (False, None)
            try:
                ok, image = self._capture.read()
            except Exception as exc:
                self._set_state(self.health.state, error=str(exc))

            if not ok or image is None:
                self.health.disconnects += 1
                self._release_capture()
                if not self._reconnect_loop():
                    return
                continue

            raw_index += 1
            self.health.frames_read += 1
            self.health.last_frame_at = self.clock()
            if self.stride > 1 and (raw_index - 1) % self.stride:
                continue

            yield Frame(
                index=emitted,
                image=_resize(image, self.max_width),
                timestamp=self.health.last_frame_at,
                # A live feed has no seekable timeline, so footage time is
                # simply how long we have been watching it.
                stream_time=self.health.last_frame_at - started,
            )
            emitted += 1
            if self.limit and emitted >= self.limit:
                return

    def _release_capture(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def release(self) -> None:
        self._stop = True
        self._release_capture()
        self._set_state(CameraState.OFFLINE)
