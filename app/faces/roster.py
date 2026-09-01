"""The roster of people a post expects to see.

The point is not to identify strangers - it is to stop alerting on the people
who are supposed to be there. A guard walking his own patrol line should not
generate the same alert as somebody climbing over it, and without a roster the
system cannot tell the difference.

**This file holds biometric data.** Three decisions follow from that, and they
are deliberate rather than incidental:

* Only face *vectors* are stored, never photographs. A vector cannot be turned
  back into a usable image of somebody's face.
* Events record the roster id and label the post chose - "SSB-114", "day
  patrol" - and never the vector. Biometrics stay in the roster file.
* Enrolment is explicit and by hand, through `tools/enrol_face.py`. Nothing
  here ever adds a person it merely saw, because a system that silently builds
  a biometric database of passers-by is a different system than this one.

Who may lawfully be enrolled, and on what basis, is not a question this file
can answer. It is a question for the deploying authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Cosine similarity above which two faces are taken to be the same person.
# SFace is usually quoted around 0.363; this is deliberately stricter, because
# waving through the wrong person costs more here than asking a known one to
# identify themselves again.
DEFAULT_THRESHOLD = 0.45


@dataclass(frozen=True)
class Person:
    """One enrolled person: an identifier, a label, and face vectors."""

    person_id: str
    label: str
    vectors: tuple[np.ndarray, ...]
    role: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        """Serialised for the roster file. Vectors, never images."""
        return {
            "person_id": self.person_id,
            "label": self.label,
            "role": self.role,
            "note": self.note,
            "vectors": [v.astype(float).tolist() for v in self.vectors],
        }

    def identity(self) -> dict:
        """What an event is allowed to carry: who, not what they look like."""
        return {"person_id": self.person_id, "label": self.label, "role": self.role}


@dataclass(frozen=True)
class Match:
    """The outcome of comparing one face against the roster."""

    person: Person | None
    similarity: float
    threshold: float

    @property
    def recognised(self) -> bool:
        return self.person is not None

    def to_dict(self) -> dict:
        return {
            "recognised": self.recognised,
            "similarity": round(self.similarity, 3),
            "threshold": self.threshold,
            "identity": self.person.identity() if self.person else None,
        }


@dataclass
class FaceRoster:
    """People this post expects to see, matched by face vector."""

    threshold: float = DEFAULT_THRESHOLD
    people: list[Person] = field(default_factory=list)
    path: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path, threshold: float = DEFAULT_THRESHOLD) -> "FaceRoster":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"face roster not found: {path}. A roster is created by "
                f"enrolment, not shipped - run tools/enrol_face.py first, "
                f"or drop --faces to run without recognition."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))

        people: list[Person] = []
        for entry in raw.get("people", []):
            vectors = tuple(
                np.asarray(v, dtype=np.float32) for v in entry.get("vectors", [])
            )
            if not vectors:
                continue
            people.append(
                Person(
                    person_id=str(entry["person_id"]),
                    label=str(entry.get("label", entry["person_id"])),
                    vectors=vectors,
                    role=str(entry.get("role", "")),
                    note=str(entry.get("note", "")),
                )
            )
        return cls(
            threshold=float(raw.get("threshold", threshold)), people=people, path=path
        )

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.path or "config/face-roster.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Face vectors of people this post expects to see. No photographs "
                        "are stored and none can be recovered from these numbers. Enrol "
                        "with tools/enrol_face.py."
                    ),
                    "threshold": self.threshold,
                    "people": [p.to_dict() for p in self.people],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return target

    # ------------------------------------------------------------- matching

    def match(self, vector: np.ndarray | None, similarity) -> Match:
        """Compare one face vector against everyone enrolled.

        `similarity` is supplied by the caller so the roster never has to own a
        model, which keeps it loadable and testable on its own.
        """
        if vector is None or not self.people:
            return Match(person=None, similarity=0.0, threshold=self.threshold)

        best, best_score = None, 0.0
        for person in self.people:
            # A person may be enrolled from several angles; any one matching is
            # a match, so the strongest wins.
            for enrolled in person.vectors:
                score = similarity(vector, enrolled)
                if score > best_score:
                    best, best_score = person, score

        if best_score < self.threshold:
            return Match(person=None, similarity=best_score, threshold=self.threshold)
        return Match(person=best, similarity=best_score, threshold=self.threshold)

    def add(self, person: Person) -> None:
        self.people = [p for p in self.people if p.person_id != person.person_id]
        self.people.append(person)

    def __len__(self) -> int:
        return len(self.people)

    def __iter__(self):
        return iter(self.people)
