"""IoU tracker with appearance-based re-identification.

Greedy IoU association holds an identity while a person stays visible. What it
cannot do is survive an occlusion: walk behind a pillar for two seconds and the
old track times out, so the person comes back as a new id with dwell reset to
zero - which quietly downgrades a loiterer to a fresh, low-risk track.

So a track that disappears is not deleted, it is *remembered*. Detections that
IoU cannot explain are compared against those remembered tracks by appearance,
and a good enough match revives the original id along with its history.

Written without SciPy or lap so a fresh checkout never pip-installs at runtime.
Swap in ByteTrack (--tracker bytetrack) for crowded scenes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from .appearance import blend, describe, similarity
from .base import Detection, iou


@dataclass
class _Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    missed: int = 0
    appearance: np.ndarray | None = None
    last_seen: float = 0.0


@dataclass
class IoUTracker:
    """Assigns stable track IDs to per-frame detections."""

    iou_threshold: float = 0.3
    max_missed: int = 15

    # Re-identification. A track that vanishes is held in a gallery for this
    # many frames, and revived if a later detection looks enough like it.
    reid_enabled: bool = True
    reid_threshold: float = 0.55
    reid_memory: int = 150          # frames a lost track stays recoverable

    _tracks: list[_Track] = field(default_factory=list, init=False)
    _lost: list[_Track] = field(default_factory=list, init=False)
    _next_id: int = field(default=1, init=False)
    _frame: int = field(default=0, init=False)
    reids: int = field(default=0, init=False)

    def update(
        self, detections: Sequence[Detection], frame: np.ndarray | None = None
    ) -> list[Detection]:
        self._frame += 1
        unmatched_tracks = list(self._tracks)
        assigned: list[tuple[int, Detection]] = []
        pending: list[tuple[int, Detection]] = []

        # Greedy: process the most confident detections first.
        order = sorted(range(len(detections)), key=lambda i: -detections[i].confidence)
        for i in order:
            det = detections[i]
            best, best_iou = None, self.iou_threshold
            for track in unmatched_tracks:
                score = iou(det.bbox, track.bbox)
                if score >= best_iou:
                    best, best_iou = track, score
            if best is None:
                # No spatial explanation; appearance gets a turn below, once
                # every confident IoU match has taken its track.
                pending.append((i, det))
                continue
            unmatched_tracks.remove(best)
            self._touch(best, det, frame)
            assigned.append((i, replace(det, track_id=best.track_id)))

        for i, det in pending:
            track = self._revive(det, frame) or self._create(det, frame)
            assigned.append((i, replace(det, track_id=track.track_id)))

        self._age(unmatched_tracks)

        # Restore the caller's original detection order.
        assigned.sort(key=lambda pair: pair[0])
        return [det for _, det in assigned]

    # ---------------------------------------------------------------- helpers

    def _touch(self, track: _Track, det: Detection, frame: np.ndarray | None) -> None:
        track.bbox = det.bbox
        track.missed = 0
        track.last_seen = self._frame
        if self.reid_enabled and frame is not None:
            track.appearance = blend(track.appearance, describe(frame, det.bbox))

    def _create(self, det: Detection, frame: np.ndarray | None) -> _Track:
        track = _Track(
            track_id=self._next_id,
            bbox=det.bbox,
            last_seen=self._frame,
            appearance=describe(frame, det.bbox) if (self.reid_enabled and frame is not None)
            else None,
        )
        self._next_id += 1
        self._tracks.append(track)
        return track

    def _revive(self, det: Detection, frame: np.ndarray | None) -> _Track | None:
        """Match an unexplained detection against recently lost tracks."""
        if not self.reid_enabled or frame is None or not self._lost:
            return None
        descriptor = describe(frame, det.bbox)
        if descriptor is None:
            return None

        best, best_score = None, self.reid_threshold
        for track in self._lost:
            score = similarity(track.appearance, descriptor)
            if score >= best_score:
                best, best_score = track, score
        if best is None:
            return None

        self._lost.remove(best)
        best.missed = 0
        best.bbox = det.bbox
        best.last_seen = self._frame
        best.appearance = blend(best.appearance, descriptor)
        self._tracks.append(best)
        self.reids += 1
        return best

    def _age(self, unmatched: list[_Track]) -> None:
        for track in unmatched:
            track.missed += 1

        still_active = []
        for track in self._tracks:
            if track.missed <= self.max_missed:
                still_active.append(track)
            elif self.reid_enabled:
                # Not gone, just out of sight: keep it recoverable.
                self._lost.append(track)
        self._tracks = still_active

        self._lost = [
            track for track in self._lost
            if self._frame - track.last_seen <= self.reid_memory and track.appearance is not None
        ]

    @property
    def lost_count(self) -> int:
        return len(self._lost)

    def reset(self) -> None:
        self._tracks.clear()
        self._lost.clear()
        self._next_id = 1
        self._frame = 0
        self.reids = 0
