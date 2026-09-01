"""Evidence clips: the seconds either side of an alert.

A still frame proves someone was inside the fence. It does not show whether
they walked in, were pushed in, or turned back immediately - and that is what
an operator has to judge, and what an incident report needs.

So every frame goes into a small rolling buffer. When an alert fires, the
buffer already holds the run-up, and the recorder keeps collecting for a few
seconds afterwards before writing one clip. The pre-roll is the part that
cannot be captured any other way: by the time the alert exists, the approach
has already happened.

Frames are kept at a reduced width, because a full-resolution buffer is
surprisingly expensive - four seconds of 960x540 is roughly 50 MB per camera,
and a post can have eight.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


@dataclass
class _PendingClip:
    """An alert whose post-roll is still being collected."""

    event_id: str
    camera_id: str
    path: Path
    frames: list[np.ndarray]
    stop_at: float


@dataclass
class ClipRecorder:
    """Keeps a rolling buffer and writes a clip around each alert."""

    directory: Path = Path("data/evidence")
    enabled: bool = True
    pre_seconds: float = 4.0
    post_seconds: float = 4.0
    fps: float = 10.0
    max_width: int = 640

    _buffer: deque = field(default_factory=deque, init=False)
    _pending: list[_PendingClip] = field(default_factory=list, init=False)
    written: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)
        self._buffer = deque(maxlen=max(1, int(self.pre_seconds * self.fps)))

    # ------------------------------------------------------------ capturing

    def observe(self, frame: np.ndarray, stream_time: float) -> None:
        """Feed every frame, alert or not. This is what makes a pre-roll possible."""
        if not self.enabled or frame is None:
            return
        small = self._shrink(frame)
        self._buffer.append((stream_time, small))
        for clip in list(self._pending):
            clip.frames.append(small)
            if stream_time >= clip.stop_at:
                self._finish(clip)

    def start(self, event_id: str, camera_id: str, when: datetime, stream_time: float) -> Path | None:
        """Begin a clip for an alert. Returns where it will be written."""
        if not self.enabled:
            return None
        name = f"{when:%Y%m%d-%H%M%S}_{camera_id}_{event_id}.mp4"
        clip = _PendingClip(
            event_id=event_id,
            camera_id=camera_id,
            path=self.directory / name,
            frames=[frame for _, frame in self._buffer],
            stop_at=stream_time + self.post_seconds,
        )
        self._pending.append(clip)
        return clip.path

    def flush(self) -> None:
        """Write out any clip still collecting, at shutdown."""
        for clip in list(self._pending):
            self._finish(clip)

    # -------------------------------------------------------------- writing

    def _finish(self, clip: _PendingClip) -> None:
        self._pending.remove(clip)
        if not clip.frames:
            return
        height, width = clip.frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(clip.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
        )
        if not writer.isOpened():
            return
        try:
            for frame in clip.frames:
                # A source that changes resolution mid-stream would otherwise
                # produce a silently corrupt file.
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
        finally:
            writer.release()
        self.written += 1

    def _shrink(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.max_width:
            return frame.copy()
        scale = self.max_width / width
        return cv2.resize(
            frame, (self.max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def buffered_frames(self) -> int:
        return len(self._buffer)
