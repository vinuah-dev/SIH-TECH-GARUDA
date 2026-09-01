"""Deterministic stand-in detector for demos and CI.

Walks one synthetic person across the frame and into the restricted zone so
the full pipeline (detect -> zone -> risk -> alert) can be exercised with no
model weights, no camera and no GPU. Useful when demonstrating on a machine
that has no sample footage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .base import Detection


@dataclass
class SimulatedPersonDetector:
    """Emits a single PERSON box that drifts left -> right each frame."""

    name: str = "SIMULATED"
    # Carries a step counter, so every camera needs its own.
    shareable: bool = False
    steps: int = 60
    walk_fraction: float = 1.0  # <1.0 walks, then stands still for the remainder
    start_x: float = 0.05
    end_x: float = 0.85
    ground_y: float = 0.90
    person_height: float = 0.45
    # >0 makes the person grow toward the lens, ending at this height fraction,
    # which is what someone walking up to tamper with the camera looks like.
    approach_camera_to: float = 0.0
    person_width: float = 0.14
    confidence: float = 0.94

    _step: int = field(default=0, init=False)

    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        height, width = frame.shape[:2]
        walk_steps = max(1, int(self.steps * self.walk_fraction))
        progress = min(1.0, self._step / max(1, walk_steps - 1))
        self._step += 1

        cx = (self.start_x + (self.end_x - self.start_x) * progress) * width
        feet_y = self.ground_y * height
        height_fraction = self.person_height
        if self.approach_camera_to > 0:
            height_fraction += (self.approach_camera_to - self.person_height) * progress
            # Someone at the lens has their feet below the frame edge.
            feet_y = height * (self.ground_y + (1.02 - self.ground_y) * progress)
        box_h = height_fraction * height
        box_w = self.person_width * width * (height_fraction / self.person_height)

        x1 = int(round(cx - box_w / 2))
        x2 = int(round(cx + box_w / 2))
        y2 = int(round(feet_y))
        y1 = int(round(feet_y - box_h))
        return [Detection(label="PERSON", confidence=self.confidence, bbox=(x1, y1, x2, y2))]

    @property
    def finished(self) -> bool:
        return self._step >= self.steps

    def reset(self) -> None:
        self._step = 0
