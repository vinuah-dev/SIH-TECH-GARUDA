"""Event persistence: append-only JSONL log plus evidence snapshots.

JSONL keeps the MVP dependency-free while staying trivially importable into
the PostgreSQL/event-store the full platform will use later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .events import IntrusionEvent


@dataclass
class EventLog:
    """Appends one JSON object per line to data/events/events-YYYYMMDD.jsonl."""

    directory: Path = Path("data/events")

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, when: datetime) -> Path:
        return self.directory / f"events-{when:%Y%m%d}.jsonl"

    def write(self, event: IntrusionEvent) -> Path:
        path = self.path_for(event.timestamp)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return path


@dataclass
class EvidenceStore:
    """Saves the annotated frame that triggered an alert.

    A still frame only - evidence *clips* need a rolling frame buffer and are
    deliberately left for the next iteration.
    """

    directory: Path = Path("data/evidence")
    enabled: bool = True

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, image: np.ndarray, event_id: str, when: datetime, camera_id: str) -> str | None:
        if not self.enabled:
            return None
        name = f"{when:%Y%m%d-%H%M%S}_{camera_id}_{event_id}.jpg"
        path = self.directory / name
        if not cv2.imwrite(str(path), image):
            return None
        return str(path)
