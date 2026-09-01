"""Following one person across several cameras.

A vehicle carries its identity on the outside: read the plate on CAM-01 and
again on CAM-03 and you know it is the same vehicle. A person carries nothing.
All that is left is what they look like, which is a far weaker signal - the
same coat is a different colour under sodium light than under infrared, and two
guards in the same uniform look alike from any distance.

So this links conservatively and says plainly that it is guessing:

* **A much higher bar than within-camera re-identification.** Recovering a
  track through a two-second occlusion on one camera is a different problem
  from claiming a match across two cameras minutes apart.
* **Overlapping sightings are never linked.** If CAM-01 can still see them
  while CAM-03 starts to, they are two people - unless the cameras genuinely
  overlap, which `min_travel_seconds = 0` allows a post to declare.
* **A link needs somewhere plausible to have walked from.** Beyond the travel
  window it is a different person who happens to own a similar coat.
* **Every link is labelled inferred**, and carries which camera actually saw
  what. An operator reading a movement history must be able to tell an
  observation from a deduction.

This exists to answer "has this person been seen elsewhere on this post?" It is
not identification, and it should never be presented as such.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from ..detection.appearance import blend, similarity

# Cross-camera appearance matching is much harder than within-camera re-ID, so
# the bar is much higher than the tracker's own 0.55.
DEFAULT_SIMILARITY = 0.72


@dataclass(frozen=True)
class Sighting:
    """One camera's view of one person."""

    camera_id: str
    track_id: int
    first_seen: float
    last_seen: float
    appearance: np.ndarray | None = field(repr=False, default=None)

    @property
    def key(self) -> tuple[str, int]:
        return (self.camera_id, self.track_id)


@dataclass
class Subject:
    """One person as the post believes it has seen them, across cameras."""

    subject_id: str
    appearance: np.ndarray | None = field(repr=False, default=None)
    sightings: list[Sighting] = field(default_factory=list)

    @property
    def cameras(self) -> tuple[str, ...]:
        seen: list[str] = []
        for sighting in self.sightings:
            if sighting.camera_id not in seen:
                seen.append(sighting.camera_id)
        return tuple(seen)

    @property
    def first_seen(self) -> float:
        return min(s.first_seen for s in self.sightings)

    @property
    def last_seen(self) -> float:
        return max(s.last_seen for s in self.sightings)

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "cameras": list(self.cameras),
            "sightings": len(self.sightings),
            "first_seen": round(self.first_seen, 1),
            "last_seen": round(self.last_seen, 1),
            # True the moment more than one camera is involved, because that
            # is precisely the part that was deduced rather than observed.
            "inferred": len(self.cameras) > 1,
        }


@dataclass
class PersonLedger:
    """Links tracked people across the cameras of one post. Thread-safe."""

    similarity_threshold: float = DEFAULT_SIMILARITY
    # How long someone may plausibly take to walk between two cameras.
    travel_window: float = 300.0
    # The shortest gap that counts as "left one camera, arrived at another".
    # Zero declares that the cameras overlap and simultaneity is allowed.
    min_travel_seconds: float = 1.0
    forget_after: float = 1800.0

    _subjects: dict[str, Subject] = field(default_factory=dict, init=False)
    _by_track: dict[tuple[str, int], str] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _next_id: int = field(default=1, init=False)
    links: int = field(default=0, init=False)

    def observe(
        self,
        camera_id: str,
        track_id: int,
        at: float,
        appearance: np.ndarray | None,
    ) -> Subject | None:
        """Record a sighting and return the subject it belongs to."""
        if track_id is None:
            return None

        with self._lock:
            key = (camera_id, track_id)
            known = self._by_track.get(key)
            if known is not None:
                return self._extend(self._subjects[known], camera_id, track_id, at, appearance)

            linked = self._find_match(camera_id, at, appearance)
            if linked is not None:
                self.links += 1
                return self._extend(linked, camera_id, track_id, at, appearance)

            subject = Subject(subject_id=f"S{self._next_id:04d}", appearance=appearance)
            self._next_id += 1
            self._subjects[subject.subject_id] = subject
            self._extend(subject, camera_id, track_id, at, appearance)
            self._prune(at)
            return subject

    # ---------------------------------------------------------------- internals

    def _extend(
        self,
        subject: Subject,
        camera_id: str,
        track_id: int,
        at: float,
        appearance: np.ndarray | None,
    ) -> Subject:
        key = (camera_id, track_id)
        self._by_track[key] = subject.subject_id
        subject.appearance = blend(subject.appearance, appearance)

        for i, sighting in enumerate(subject.sightings):
            if sighting.key == key:
                subject.sightings[i] = Sighting(
                    camera_id=camera_id,
                    track_id=track_id,
                    first_seen=sighting.first_seen,
                    last_seen=max(sighting.last_seen, at),
                    appearance=appearance if appearance is not None else sighting.appearance,
                )
                return subject

        subject.sightings.append(
            Sighting(camera_id, track_id, first_seen=at, last_seen=at, appearance=appearance)
        )
        return subject

    def _find_match(
        self, camera_id: str, at: float, appearance: np.ndarray | None
    ) -> Subject | None:
        if appearance is None:
            return None

        best, best_score = None, self.similarity_threshold
        for subject in self._subjects.values():
            if camera_id in subject.cameras:
                # This camera has already given this subject a track; a second
                # one is two people, not the same person twice.
                continue
            gap = at - subject.last_seen
            if gap < self.min_travel_seconds:
                # Still visible elsewhere, so not the same person walking here.
                continue
            if gap > self.travel_window:
                continue
            score = similarity(subject.appearance, appearance)
            if score >= best_score:
                best, best_score = subject, score
        return best

    def _prune(self, now: float) -> None:
        stale = [
            sid for sid, subject in self._subjects.items()
            if subject.sightings and now - subject.last_seen > self.forget_after
        ]
        for sid in stale:
            subject = self._subjects.pop(sid)
            for sighting in subject.sightings:
                self._by_track.pop(sighting.key, None)

    # ---------------------------------------------------------------- queries

    def subject_for(self, camera_id: str, track_id: int | None) -> Subject | None:
        if track_id is None:
            return None
        with self._lock:
            sid = self._by_track.get((camera_id, track_id))
            return self._subjects.get(sid) if sid else None

    def seen_on_multiple_cameras(self) -> list[Subject]:
        with self._lock:
            return [s for s in self._subjects.values() if len(s.cameras) > 1]

    @property
    def subjects(self) -> int:
        with self._lock:
            return len(self._subjects)

    def reset(self) -> None:
        with self._lock:
            self._subjects.clear()
            self._by_track.clear()
            self._next_id = 1
            self.links = 0
