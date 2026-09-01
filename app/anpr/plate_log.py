"""A record of every vehicle that passed, not only the ones that alarmed.

Alerts answer "what went wrong". A post also needs the other question: *what
came through here today?* A vehicle that crossed nothing and broke no rule is
still a vehicle that passed the post, and a week later the only thing anyone
wants is the list.

Two things make that list usable rather than a wall of noise:

**One entry per vehicle, not per frame.** A vehicle is in view for hundreds of
frames. Logging each one buries the day in duplicates of a single car.

**But a vehicle that comes back is a second entry.** That is the whole point of
the log - a car that passes at 09:00 and again at 09:02 is exactly the pattern
worth seeing. So duplicates are suppressed only inside a short window; past it,
the same plate is a genuine second crossing.

The entry is written when the vehicle leaves, not when it arrives. A plate is
sharpest somewhere in the middle of a pass, and waiting costs nothing - the
vehicle is already gone by the time anyone reads the log.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..detection.base import Detection
from ..detection.classes import is_vehicle
from .plate import sharpness

# How long the same vehicle stays suppressed. Ten seconds is long enough to
# cover one pass through a camera's field of view and short enough that a
# vehicle which turns around and comes back is recorded as a second crossing.
DEFAULT_WINDOW = 10.0

# A track unseen for this long has left. Short enough to write the entry while
# the pass is still recent, long enough to survive a brief occlusion behind
# another vehicle rather than splitting one pass into two entries.
DEFAULT_LINGER = 2.0


@dataclass
class _Pending:
    """A vehicle currently in view, accumulating its best evidence."""

    camera_id: str
    track_id: int
    label: str
    first_seen: float
    last_seen: float
    # Epoch seconds, the same clock Frame.timestamp carries. Converted to a
    # datetime only where one is needed, so there is one clock in this file
    # rather than two that can disagree.
    first_at: float
    frames: int = 1
    crop: np.ndarray | None = field(default=None, repr=False)
    best_sharpness: float = 0.0
    plate: str | None = None
    confidence: float = 0.0
    display: str | None = None


@dataclass
class PlateLog:
    """Every vehicle that passed, once each. Thread-safe across a fleet."""

    directory: Path | None = None
    evidence: object | None = None          # EvidenceStore, when crops are wanted
    window: float = DEFAULT_WINDOW
    linger: float = DEFAULT_LINGER
    enabled: bool = True

    _pending: dict[tuple[str, int], _Pending] = field(default_factory=dict, init=False)
    _recent: dict[str, float] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    written: int = field(default=0, init=False)
    suppressed: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.directory is not None:
            self.directory = Path(self.directory)
            self.directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ input

    def observe(
        self,
        camera_id: str,
        detection: Detection,
        at: float,
        when: float,
        crop: np.ndarray | None = None,
        reading=None,
    ) -> None:
        """Note a vehicle in view, keeping the sharpest plate crop seen so far."""
        if not self.enabled or detection.track_id is None:
            return
        if not is_vehicle(detection.label):
            return

        key = (camera_id, detection.track_id)
        with self._lock:
            entry = self._pending.get(key)
            if entry is None:
                entry = _Pending(
                    camera_id=camera_id,
                    track_id=detection.track_id,
                    label=detection.label,
                    first_seen=at,
                    last_seen=at,
                    first_at=when,
                )
                self._pending[key] = entry
            else:
                entry.last_seen = at
                entry.frames += 1

            if crop is not None and crop.size:
                focus = sharpness(crop)
                if entry.crop is None or focus > entry.best_sharpness:
                    entry.crop, entry.best_sharpness = crop, focus

            if reading is not None and getattr(reading, "text", None):
                if reading.confidence >= entry.confidence:
                    entry.plate = reading.text
                    entry.confidence = reading.confidence
                    entry.display = getattr(reading, "display", None) or reading.text

    # ----------------------------------------------------------------- output

    def flush_departed(self, now: float) -> list[dict]:
        """Write an entry for every vehicle that has left the frame."""
        if not self.enabled:
            return []
        with self._lock:
            gone = [
                key for key, entry in self._pending.items()
                if now - entry.last_seen > self.linger
            ]
            leaving = [self._pending.pop(key) for key in gone]
        return [row for row in (self._write(entry) for entry in leaving) if row]

    def close(self) -> list[dict]:
        """Write everything still in view. The run is over; they have all left."""
        if not self.enabled:
            return []
        with self._lock:
            leaving = list(self._pending.values())
            self._pending.clear()
        return [row for row in (self._write(entry) for entry in leaving) if row]

    # -------------------------------------------------------------- internals

    def _write(self, entry: _Pending) -> dict | None:
        # A read plate identifies the vehicle far better than a track id, which
        # is why it is preferred as the key: one vehicle whose track split in
        # two behind a bus is still one crossing.
        key = entry.plate or f"{entry.camera_id}#{entry.track_id}"
        with self._lock:
            seen_at = self._recent.get(key)
            if seen_at is not None and entry.last_seen - seen_at < self.window:
                self.suppressed += 1
                return None
            self._recent[key] = entry.last_seen
            self._forget(entry.last_seen)

        path = None
        if entry.crop is not None and self.evidence is not None:
            path = self.evidence.save(
                entry.crop,
                f"pass-{entry.camera_id}-{entry.track_id}",
                datetime.fromtimestamp(entry.first_at),
                entry.camera_id,
            )

        row = {
            "at": datetime.fromtimestamp(entry.first_at).isoformat(timespec="seconds"),
            "camera_id": entry.camera_id,
            "track_id": entry.track_id,
            "vehicle": entry.label,
            "plate": entry.plate,
            "display": entry.display,
            "confidence": round(entry.confidence, 3) if entry.plate else None,
            # Stated plainly, because a log of vehicles whose plates were mostly
            # not read must not be mistaken for a log of plates.
            "plate_read": entry.plate is not None,
            "seconds_in_view": round(entry.last_seen - entry.first_seen, 1),
            "frames": entry.frames,
            "crop": path,
        }
        self._append(row, entry.first_at)
        with self._lock:
            self.written += 1
        return row

    def _append(self, row: dict, when: float) -> None:
        if self.directory is None:
            return
        day = datetime.fromtimestamp(when)
        path = self.directory / f"passes-{day:%Y%m%d}.jsonl"
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

    def _forget(self, now: float) -> None:
        """Drop suppression keys past their window, so the dict cannot grow forever."""
        stale = [k for k, at in self._recent.items() if now - at > self.window * 6]
        for key in stale:
            self._recent.pop(key, None)
