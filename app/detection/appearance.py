"""Appearance descriptors, for recognising the same person again.

An IoU tracker only knows where a box was last frame. Walk behind a pillar and
come back, and it hands out a fresh id - which resets dwell to zero and turns a
loiterer into a brand new, low-risk track. That is the weakest link under every
behaviour in this system.

The fix is to remember what someone *looked* like. This module keeps that
deliberately cheap: a hue/saturation histogram of the torso region, which needs
no extra dependency and no second model on the CPU budget. It is not a deep
re-identification network - it will not tell two people in the same uniform
apart - but it reliably survives the few seconds of occlusion that break plain
IoU tracking, which is the case that actually costs us alerts.
"""

from __future__ import annotations

import cv2
import numpy as np

# Hue matters most for clothing; saturation separates a dark coat from a bright
# one. Value is left out on purpose, because it swings with lighting and shadow.
HUE_BINS = 24
SAT_BINS = 8

# Pixels too dark or too washed out have meaningless hue, so they are masked
# out rather than allowed to dominate the histogram.
MIN_SATURATION = 40
MIN_VALUE = 40


def torso_box(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The middle of the upper body: mostly clothing, least background.

    A full person box is largely background at the shoulders and legs, so
    describing all of it mostly describes the scene, not the person.
    """
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return (
        x1 + int(width * 0.20),
        y1 + int(height * 0.15),
        x2 - int(width * 0.20),
        y1 + int(height * 0.55),
    )


def body_box(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The middle of a vehicle: bodywork rather than glass, sky or road.

    A vehicle has no torso, and its upper region is mostly windscreen - which
    reflects whatever is around it and describes the scene, not the vehicle.
    """
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return (
        x1 + int(width * 0.15),
        y1 + int(height * 0.35),
        x2 - int(width * 0.15),
        y2 - int(height * 0.10),
    )


def describe(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    region=None,
) -> np.ndarray | None:
    """Build a normalized appearance descriptor, or None if there is too little to go on."""
    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (region or torso_box)(bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 1] >= MIN_SATURATION) & (hsv[:, :, 2] >= MIN_VALUE)
    ).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return None

    histogram = cv2.calcHist([hsv], [0, 1], mask, [HUE_BINS, SAT_BINS], [0, 180, 0, 256])
    total = float(histogram.sum())
    if total <= 0:
        return None
    return (histogram / total).flatten().astype(np.float32)


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """How alike two descriptors are, from 0.0 to 1.0.

    Histogram intersection: the fraction of the distribution the two share.
    It is bounded, cheap, and behaves sensibly when part of the person is
    occluded, which cosine distance does not.
    """
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float(np.minimum(a, b).sum())


def blend(existing: np.ndarray | None, latest: np.ndarray | None, weight: float = 0.3
          ) -> np.ndarray | None:
    """Nudge a track's stored appearance toward what we just saw.

    Averaging over time absorbs a single bad crop - a half-occluded frame, a
    moment of odd lighting - without letting the track drift onto whatever
    happens to be behind the person.
    """
    if latest is None:
        return existing
    if existing is None:
        return latest
    merged = (1.0 - weight) * existing + weight * latest
    total = float(merged.sum())
    return (merged / total).astype(np.float32) if total > 0 else existing
