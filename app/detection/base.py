"""Shared detection types.

Everything downstream (zones, risk, alerts) depends only on this module, so
the detector backend can be swapped without touching the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected object in one frame, in pixel coordinates."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: int | None = None

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def ground_point(self) -> tuple[int, int]:
        """Bottom-centre of the box - where the person's feet meet the ground.

        Zone membership is tested here rather than at the centroid: a standing
        person's centroid sits at torso height and crosses a ground-plane fence
        line noticeably before their feet actually do.
        """
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) // 2, y2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def track_label(self) -> str:
        return f"P{self.track_id:03d}" if self.track_id is not None else "P---"


class Detector(Protocol):
    """Interface every detection backend implements.

    `shareable` says whether one instance may serve several cameras at once.
    A backend that carries per-stream state - a tracker holding track ids, or
    a scripted simulator counting steps - must answer False, or two cameras
    would corrupt each other's state.
    """

    name: str
    shareable: bool

    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        ...


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
