"""Fleet-wide plate ledger: one camera reads what another could not.

A border post has cameras, plural, and a vehicle passes several of them. Any
one camera gets a bad angle, a blurred frame or a plate obscured by a tow bar -
but rarely do all of them. Keeping plate reads per camera throws that away.

This ledger is shared by every camera in a fleet and does two things:

**It pools the votes.** A weak read on CAM-01 and a weak read on CAM-03 for the
same number are, together, a confident read - even though neither camera alone
would have reported it.

**It hands a plate back to the camera that missed it.** A vehicle seen but not
read on CAM-01 is remembered by appearance. When CAM-03 reads a plate off a
vehicle that looks the same, within a plausible travel window, the earlier
sighting is credited with that number.

That second step is inference, not observation, so it is labelled as such:
a resolved plate always records which camera actually read it.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..detection.appearance import similarity

# Below this, two vehicle crops are not the same vehicle.
DEFAULT_HANDOFF_SIMILARITY = 0.55


@dataclass(frozen=True)
class Sighting:
    """One camera's view of a vehicle whose plate it could not read."""

    camera_id: str
    track_id: int
    at: float
    appearance: np.ndarray | None
    label: str = "VEHICLE"

    @property
    def key(self) -> tuple[str, int]:
        return (self.camera_id, self.track_id)


@dataclass(frozen=True)
class LedgerPlate:
    """What the fleet collectively knows about one vehicle's plate."""

    text: str
    confidence: float
    reads: int
    cameras: tuple[str, ...]
    read_by: str
    inferred: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "reads": self.reads,
            "cameras": list(self.cameras),
            "read_by": self.read_by,
            "inferred": self.inferred,
        }


@dataclass
class _PlateRecord:
    votes: Counter = field(default_factory=Counter)
    confidence: float = 0.0
    cameras: list[str] = field(default_factory=list)
    first_read_by: str = ""
    last_seen: float = 0.0


@dataclass
class PlateLedger:
    """Plate reads pooled across every camera in a fleet. Thread-safe."""

    handoff_similarity: float = DEFAULT_HANDOFF_SIMILARITY
    handoff_window: float = 180.0     # seconds a vehicle may take between cameras
    forget_after: float = 900.0

    _plates: dict[str, _PlateRecord] = field(default_factory=dict, init=False)
    # Which plate each (camera, track) is now credited with.
    _assigned: dict[tuple[str, int], LedgerPlate] = field(default_factory=dict, init=False)
    _unread: list[Sighting] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    handoffs: int = field(default=0, init=False)

    # ----------------------------------------------------------- recording

    def note_sighting(self, sighting: Sighting) -> None:
        """Remember a vehicle whose plate has not been read yet."""
        with self._lock:
            if sighting.key in self._assigned:
                return
            self._unread = [s for s in self._unread if s.key != sighting.key]
            self._unread.append(sighting)
            self._prune(sighting.at)

    def record_read(
        self,
        camera_id: str,
        track_id: int,
        plate: str,
        confidence: float,
        at: float,
        appearance: np.ndarray | None = None,
    ) -> LedgerPlate:
        """Log a plate this camera actually read, and share it with the fleet."""
        with self._lock:
            record = self._plates.setdefault(plate, _PlateRecord())
            record.votes[camera_id] += 1
            record.confidence = max(record.confidence, confidence)
            record.last_seen = at
            if camera_id not in record.cameras:
                record.cameras.append(camera_id)
            if not record.first_read_by:
                record.first_read_by = camera_id

            resolved = self._build(plate, record, read_by=camera_id, inferred=False)
            self._assigned[(camera_id, track_id)] = resolved
            self._unread = [s for s in self._unread if s.key != (camera_id, track_id)]
            self._hand_back(plate, record, camera_id, at, appearance)
            self._prune(at)
            return resolved

    def _hand_back(
        self,
        plate: str,
        record: _PlateRecord,
        camera_id: str,
        at: float,
        appearance: np.ndarray | None,
    ) -> None:
        """Credit earlier, unread sightings of the same vehicle with this plate."""
        if appearance is None:
            return
        for sighting in list(self._unread):
            if sighting.camera_id == camera_id:
                continue  # the same camera would have used its own read
            if abs(at - sighting.at) > self.handoff_window:
                continue
            if similarity(sighting.appearance, appearance) < self.handoff_similarity:
                continue

            self._assigned[sighting.key] = self._build(
                plate, record, read_by=camera_id, inferred=True
            )
            self._unread.remove(sighting)
            self.handoffs += 1

    @staticmethod
    def _build(plate: str, record: _PlateRecord, read_by: str, inferred: bool) -> LedgerPlate:
        return LedgerPlate(
            text=plate,
            confidence=record.confidence,
            reads=sum(record.votes.values()),
            cameras=tuple(record.cameras),
            read_by=read_by,
            inferred=inferred,
        )

    # ------------------------------------------------------------ retrieval

    def resolve(self, camera_id: str, track_id: int | None) -> LedgerPlate | None:
        """The plate for one camera's track, whether it read it or not."""
        if track_id is None:
            return None
        with self._lock:
            return self._assigned.get((camera_id, track_id))

    def cameras_for(self, plate: str) -> tuple[str, ...]:
        with self._lock:
            record = self._plates.get(plate)
            return tuple(record.cameras) if record else ()

    def known_plates(self) -> dict[str, int]:
        with self._lock:
            return {p: sum(r.votes.values()) for p, r in self._plates.items()}

    @property
    def pending_sightings(self) -> int:
        with self._lock:
            return len(self._unread)

    # ------------------------------------------------------------- upkeep

    def _prune(self, now: float) -> None:
        self._unread = [s for s in self._unread if now - s.at <= self.handoff_window]
        stale = [p for p, r in self._plates.items() if now - r.last_seen > self.forget_after]
        for plate in stale:
            del self._plates[plate]

    def reset(self) -> None:
        with self._lock:
            self._plates.clear()
            self._assigned.clear()
            self._unread.clear()
            self.handoffs = 0
