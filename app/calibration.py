"""Ground-plane calibration.

A camera sees a perspective image; a border post cares about metres on the
ground. This module maps one to the other with a homography built from four
reference points whose real-world positions are known.

Why it matters: without it every threshold in the system is expressed in
*frame widths*, which means the same number means something different on a
camera watching a 200 m approach and one covering a 10 m gate. With it,
"moving at 1.4 m/s, 12 m from the fence" is comparable across cameras.

Calibration is entirely optional. Uncalibrated cameras keep working exactly as
before, reporting normalized frame units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

Point = tuple[float, float]

# A homography needs exactly four correspondences to be determined.
REQUIRED_POINTS = 4


@dataclass(frozen=True)
class GroundPlane:
    """Maps normalized image coordinates to ground-plane metres."""

    matrix: np.ndarray
    unit: str = "m"

    @classmethod
    def from_points(cls, pairs: Sequence[tuple[Point, Point]], unit: str = "m") -> "GroundPlane":
        """Build from four (image, world) correspondences.

        `image` points are normalized 0..1, matching the zone convention.
        `world` points are real-world coordinates on the ground plane, in
        whatever unit the operator surveyed with (metres by default).
        """
        if len(pairs) != REQUIRED_POINTS:
            raise ValueError(
                f"ground-plane calibration needs exactly {REQUIRED_POINTS} points, "
                f"got {len(pairs)}"
            )
        image = np.float32([p[0] for p in pairs])
        world = np.float32([p[1] for p in pairs])

        for name, points in (("image", image), ("world", world)):
            if len(np.unique(points, axis=0)) != REQUIRED_POINTS:
                raise ValueError(f"{name} calibration points must all be distinct")

        matrix = cv2.getPerspectiveTransform(image, world)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("calibration points are degenerate (three or more collinear)")
        return cls(matrix=matrix, unit=unit)

    @classmethod
    def from_config(cls, raw: dict) -> "GroundPlane":
        points = raw.get("points", [])
        pairs = []
        for entry in points:
            image = entry["image"]
            world = entry["world"]
            if len(image) != 2 or len(world) != 2:
                raise ValueError("each calibration point needs 2-value image and world entries")
            pairs.append(((float(image[0]), float(image[1])), (float(world[0]), float(world[1]))))
        return cls.from_points(pairs, unit=str(raw.get("unit", "m")))

    def project(self, point: Point) -> Point:
        """Normalized image point -> ground-plane coordinates."""
        source = np.float32([[[point[0], point[1]]]])
        projected = cv2.perspectiveTransform(source, self.matrix)[0][0]
        return (float(projected[0]), float(projected[1]))

    def distance(self, a: Point, b: Point) -> float:
        """Real-world distance between two image points on the ground plane."""
        ax, ay = self.project(a)
        bx, by = self.project(b)
        return float(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
