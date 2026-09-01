"""Video ingestion.

One factory (`open_source`) resolves a CLI string into an iterable of frames,
so the pipeline never cares whether footage comes from a file, a webcam, a
still image or the built-in simulator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Frame:
    """One frame plus its two clocks.

    `timestamp` is wall-clock, used for the event record and the day/night
    window. `stream_time` is seconds elapsed *within the footage*, and is what
    speed, dwell and cooldown must be measured against - otherwise a file
    replayed at 500 fps would report absurd speeds and a cooldown that covers
    the wrong span of footage.
    """

    index: int
    image: np.ndarray
    timestamp: float
    stream_time: float = 0.0

    @property
    def size(self) -> tuple[int, int]:
        height, width = self.image.shape[:2]
        return width, height


class FrameSource(Protocol):
    name: str
    kind: str
    fps: float

    def frames(self) -> Iterator[Frame]:
        ...

    def release(self) -> None:
        ...


def _resize(image: np.ndarray, max_width: int | None) -> np.ndarray:
    if not max_width:
        return image
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    return cv2.resize(image, (max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)


@dataclass
class ImageSource:
    """A single still image, replayed `repeat` times."""

    path: Path
    repeat: int = 1
    max_width: int | None = 960
    kind: str = "IMAGE"
    fps: float = 0.0

    def __post_init__(self) -> None:
        self.name = str(self.path)

    def frames(self) -> Iterator[Frame]:
        image = cv2.imread(str(self.path))
        if image is None:
            raise RuntimeError(f"could not decode image: {self.path}")
        image = _resize(image, self.max_width)
        for i in range(max(1, self.repeat)):
            yield Frame(
                index=i, image=image.copy(), timestamp=time.time(), stream_time=float(i)
            )

    def release(self) -> None:
        return None


@dataclass
class VideoSource:
    """A video file or a live camera device."""

    target: str | int
    kind: str = "VIDEO"
    stride: int = 1
    limit: int | None = None
    loop: bool = False
    max_width: int | None = 960

    def __post_init__(self) -> None:
        self.name = str(self.target)
        self._capture = cv2.VideoCapture(self.target)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video source: {self.target}")
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS)) or 0.0
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def _stream_time(self, raw_index: int, wall_start: float) -> float:
        """Seconds into the footage.

        A live camera has no meaningful presentation timestamp, so it falls
        back to elapsed wall time - which for a live feed is the same thing.
        """
        if self.kind == "WEBCAM":
            return time.time() - wall_start
        position = float(self._capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if position > 0:
            return position / 1000.0
        # Some containers report no PTS; fall back to the nominal frame rate.
        return (raw_index - 1) / self.fps if self.fps > 0 else float(raw_index - 1)

    def frames(self) -> Iterator[Frame]:
        raw_index = 0
        emitted = 0
        wall_start = time.time()
        loops = 0
        while True:
            ok, image = self._capture.read()
            if not ok:
                if self.loop and self.frame_count > 0:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    loops += 1
                    continue
                break
            raw_index += 1
            if self.stride > 1 and (raw_index - 1) % self.stride:
                continue
            stream_time = self._stream_time(raw_index, wall_start)
            if loops and self.frame_count > 0 and self.fps > 0:
                # Keep the clock monotonic across replays.
                stream_time += loops * (self.frame_count / self.fps)
            yield Frame(
                index=emitted,
                image=_resize(image, self.max_width),
                timestamp=time.time(),
                stream_time=stream_time,
            )
            emitted += 1
            if self.limit and emitted >= self.limit:
                break

    def release(self) -> None:
        self._capture.release()


@dataclass
class SyntheticSource:
    """Generated frames for the model-free demo path.

    Renders a plain night-time border scene; detections are drawn on top by
    the annotator, so this stays decoupled from the simulated detector.
    """

    frames_to_emit: int = 60
    width: int = 960
    height: int = 540
    kind: str = "SYNTHETIC"
    fps: float = 15.0
    name: str = "synthetic"

    def _scene(self) -> np.ndarray:
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        horizon = int(self.height * 0.30)
        # Sky: dark blue gradient. Ground: flat dark earth.
        for y in range(horizon):
            shade = int(18 + 22 * (y / max(1, horizon)))
            image[y, :] = (shade + 14, shade + 6, shade)
        image[horizon:, :] = (28, 32, 30)
        cv2.line(image, (0, horizon), (self.width, horizon), (70, 76, 72), 1)
        return image

    def frames(self) -> Iterator[Frame]:
        base = self._scene()
        for i in range(self.frames_to_emit):
            yield Frame(
                index=i,
                image=base.copy(),
                timestamp=time.time(),
                stream_time=i / self.fps if self.fps else float(i),
            )

    def release(self) -> None:
        return None


def open_source(
    spec: str,
    *,
    stride: int = 1,
    limit: int | None = None,
    loop: bool = False,
    max_width: int | None = 960,
    synthetic_frames: int = 60,
    reconnect_attempts: int = 0,
) -> FrameSource:
    """Resolve a --source string into a frame source.

    Accepted forms: `synthetic`, `webcam[:N]`, a plain integer device index,
    an `rtsp://` / `http://` stream URL, an image path, or a video path.
    """
    spec = spec.strip()

    if spec.lower() == "synthetic":
        return SyntheticSource(frames_to_emit=synthetic_frames)

    # Imported here because stream.py imports from this module.
    from .stream import StreamSource, is_network_source

    if is_network_source(spec):
        scheme = spec.split("://", 1)[0].upper()
        return StreamSource(
            url=spec,
            kind="RTSP" if scheme.startswith("RTSP") else scheme,
            stride=stride,
            limit=limit,
            max_width=max_width,
            reconnect_attempts=reconnect_attempts,
        )

    if spec.lower().startswith("webcam"):
        _, _, index = spec.partition(":")
        device = int(index) if index.strip().isdigit() else 0
        return VideoSource(device, kind="WEBCAM", stride=stride, limit=limit, max_width=max_width)

    if spec.isdigit():
        return VideoSource(int(spec), kind="WEBCAM", stride=stride, limit=limit, max_width=max_width)

    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"source not found: {spec}")
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return ImageSource(path, repeat=max(1, limit or 1), max_width=max_width)
    return VideoSource(
        str(path), kind="VIDEO", stride=stride, limit=limit, loop=loop, max_width=max_width
    )
