"""Recognising the people a post expects to see.

The purpose is subtraction, not identification. A guard walking his own patrol
line trips exactly the same virtual fence as somebody climbing over it, and an
operator who is shown both learns to ignore both. Recognising the guard is what
lets the other alert mean something.

Two asymmetries are deliberate, and they are the whole design:

**Being recognised lowers risk. Not being recognised raises nothing.** Most
people at a border are not enrolled and never will be, so treating an unknown
face as evidence would flag the entire population. Absence from a roster is not
evidence of anything - it is the normal case. A post where everyone genuinely
should be enrolled can turn that on with `unrecognised_points`.

**A face is only trusted when it is large enough to mean something.** Below
about 60 pixels across, the vector is noise, and matching noise against a
roster produces confident nonsense - the same failure ANPR spent a day
learning. Small faces are ignored rather than guessed at.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..detection.base import Detection
from ..detection.classes import is_person
from .detector import FaceEngine
from .roster import FaceRoster, Match


@dataclass
class _TrackFace:
    """What has been seen of one tracked person's face."""

    votes: Counter = field(default_factory=Counter)
    best: Match | None = None
    attempts: int = 0
    last_attempt_frame: int = -999
    largest_face: int = 0


@dataclass
class FaceRecognitionEngine:
    """Matches tracked people against the roster, frame by frame."""

    enabled: bool = True
    faces: FaceEngine | None = None
    roster: FaceRoster | None = None

    # Recognition is worth a forward pass only now and then: a person's face
    # does not change between frames, and the roster answer will not either.
    attempt_interval: int = 8
    max_attempts: int = 10
    # Agreeing sightings before an identity is acted on. One match is a guess.
    min_agreeing: int = 2

    _tracks: dict[int, _TrackFace] = field(default_factory=dict, init=False)
    _frame: int = field(default=0, init=False)
    attempts_made: int = field(default=0, init=False)

    @property
    def available(self) -> bool:
        return bool(
            self.enabled
            and self.faces is not None
            and self.faces.available
            and self.roster is not None
            and len(self.roster)
        )

    def tick(self) -> None:
        self._frame += 1

    def observe(self, frame: np.ndarray, detection: Detection) -> Match | None:
        """Consider one tracked person. Returns the identity known so far."""
        if not self.available or frame is None or detection.track_id is None:
            return None
        if not is_person(detection.label):
            return None

        state = self._tracks.setdefault(detection.track_id, _TrackFace())
        if self._should_attempt(state):
            self._attempt(frame, detection, state)
        return self._settled(state)

    def _should_attempt(self, state: _TrackFace) -> bool:
        if state.attempts >= self.max_attempts:
            return False
        if self._frame - state.last_attempt_frame < self.attempt_interval:
            return False
        # Once enough sightings agree, stop spending passes on a settled answer.
        settled = self._settled(state)
        return settled is None or not settled.recognised

    def _attempt(self, frame: np.ndarray, detection: Detection, state: _TrackFace) -> None:
        state.attempts += 1
        state.last_attempt_frame = self._frame
        self.attempts_made += 1

        found = self.faces.detect_on(frame, detection.bbox)
        if not found:
            return
        face = found[0]
        state.largest_face = max(state.largest_face, face.width)

        vector = self.faces.embed(frame, face)
        if vector is None:
            return

        match = self.roster.match(vector, self.faces.similarity)
        if match.recognised:
            state.votes[match.person.person_id] += 1
            if state.best is None or match.similarity > state.best.similarity:
                state.best = match

    def _settled(self, state: _TrackFace) -> Match | None:
        """The identity only once enough sightings agree on it."""
        if state.best is None or not state.votes:
            return None
        person_id, count = state.votes.most_common(1)[0]
        if count < self.min_agreeing:
            return None
        if state.best.person is None or state.best.person.person_id != person_id:
            return None
        return state.best

    def identity_for(self, track_id: int | None) -> Match | None:
        if track_id is None:
            return None
        state = self._tracks.get(track_id)
        return self._settled(state) if state else None

    def forget(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def reset(self) -> None:
        self._tracks.clear()
        self._frame = 0
        self.attempts_made = 0
