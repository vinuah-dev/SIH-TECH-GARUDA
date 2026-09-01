"""Face detection and face embeddings, using OpenCV's own models.

YuNet detects, SFace turns an aligned face into a 128-number vector that can be
compared to another. Both ship with OpenCV, so this adds no dependency - only
two model files, fetched once by `tools/fetch_face_models.py`.

A hard limit worth stating before anything else: **a face needs roughly 80
pixels across to be recognised at all.** At the ranges a perimeter camera works
at, a person 200 px tall has a face of about 25 px, which is nowhere near
enough. Face recognition here is a *gate and checkpoint* capability - a camera
looking at people arriving at a post - and not something a fence-line camera
can do. `tools/face_envelope.py` measures where the limit actually falls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

DETECTOR_WEIGHTS = Path("models/face_detection_yunet.onnx")
RECOGNISER_WEIGHTS = Path("models/face_recognition_sface.onnx")

# Below this a face carries too little detail for the embedding to mean
# anything, and matching it against a roster is worse than not trying.
MIN_FACE_WIDTH = 60


@dataclass(frozen=True)
class Face:
    """One detected face, in full-frame coordinates."""

    bbox: tuple[int, int, int, int]
    confidence: float
    # The raw YuNet row, kept because alignment needs its landmarks.
    row: np.ndarray = field(repr=False, default=None)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class FaceEngine:
    """Finds faces and turns them into comparable vectors."""

    detector_weights: Path = DETECTOR_WEIGHTS
    recogniser_weights: Path = RECOGNISER_WEIGHTS
    confidence: float = 0.75
    nms: float = 0.3
    min_width: int = MIN_FACE_WIDTH

    detector: object | None = field(default=None, init=False)
    recogniser: object | None = field(default=None, init=False)
    available: bool = field(default=False, init=False)
    last_error: str | None = field(default=None, init=False)
    _size: tuple[int, int] = field(default=(0, 0), init=False)

    def __post_init__(self) -> None:
        self.detector_weights = Path(self.detector_weights)
        self.recogniser_weights = Path(self.recogniser_weights)
        missing = [p for p in (self.detector_weights, self.recogniser_weights)
                   if not p.exists()]
        if missing:
            self.last_error = f"missing {', '.join(str(p) for p in missing)}"
            return
        try:
            self.detector = cv2.FaceDetectorYN.create(
                str(self.detector_weights), "", (320, 320),
                self.confidence, self.nms, 5000,
            )
            self.recogniser = cv2.FaceRecognizerSF.create(
                str(self.recogniser_weights), ""
            )
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on the build
            self.last_error = str(exc)

    # ------------------------------------------------------------ detection

    def detect(self, frame: np.ndarray) -> Sequence[Face]:
        """Every face in the frame, largest first."""
        if not self.available or frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        if (width, height) != self._size:
            self.detector.setInputSize((width, height))
            self._size = (width, height)

        try:
            _, rows = self.detector.detect(frame)
        except Exception as exc:
            self.last_error = str(exc)
            return []
        if rows is None:
            return []

        faces: list[Face] = []
        for row in rows:
            x, y, w, h = (int(round(v)) for v in row[:4])
            if w < self.min_width:
                continue
            faces.append(
                Face(
                    bbox=(max(0, x), max(0, y), min(width, x + w), min(height, y + h)),
                    confidence=float(row[-1]),
                    row=row,
                )
            )
        faces.sort(key=lambda f: -f.width)
        return faces

    def detect_on(
        self, frame: np.ndarray, person_bbox: tuple[int, int, int, int]
    ) -> Sequence[Face]:
        """Faces belonging to one person box.

        Detection runs on the whole frame and results are attributed
        afterwards, so several people in one frame share a single pass.
        """
        px1, py1, px2, py2 = person_bbox
        # A face sits in the upper part of a person box; anything lower is
        # somebody else's head that happens to overlap.
        cutoff = py1 + (py2 - py1) * 0.5
        inside = []
        for face in self.detect(frame):
            fx1, fy1, fx2, fy2 = face.bbox
            cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            if px1 <= cx <= px2 and py1 <= cy <= max(cutoff, py1 + 1):
                inside.append(face)
        return inside

    # ------------------------------------------------------------ embedding

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray | None:
        """A 128-number vector for one face, or None if it cannot be made."""
        if not self.available or face.row is None:
            return None
        try:
            aligned = self.recogniser.alignCrop(frame, face.row)
            return self.recogniser.feature(aligned).flatten().astype(np.float32)
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def similarity(self, a: np.ndarray | None, b: np.ndarray | None) -> float:
        """Cosine similarity between two face vectors, 0.0 to 1.0."""
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        try:
            return float(
                self.recogniser.match(
                    a.reshape(1, -1), b.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE
                )
            )
        except Exception:
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / denominator) if denominator else 0.0
