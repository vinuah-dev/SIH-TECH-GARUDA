"""Vehicle registry lookup, and catching plates that do not belong.

A plate is only evidence if it matches the vehicle carrying it. Cloned and
transplanted plates are a standard way to move a vehicle that would otherwise
be stopped, and they are detectable without any extra sensor: read the plate,
look up what is registered against it, and compare that to what the camera can
plainly see. A plate registered to a motorcycle, bolted to a truck, is worth an
operator's attention on its own.

**This does not talk to VAHAN or any RTO system.** It reads a local registry
file - which is what a border post would actually be issued, as a periodic
extract or a watchlist. `RegistryBackend` is the seam a real authorised API
would slot into later; nothing above this module would change.

Privacy note: registry extracts carry owner details. Only the vehicle
attributes needed for the comparison are copied into an event; owner
information stays in the registry and out of the event log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..detection.classes import VEHICLE, category_of

# RTO paperwork and COCO do not use the same words for the same thing.
CLASS_ALIASES = {
    "MOTORCYCLE": {"MOTORCYCLE", "MOTOR CYCLE", "M-CYCLE", "MCWG", "SCOOTER", "TWO WHEELER", "2W"},
    "CAR": {"CAR", "MOTOR CAR", "LMV", "LMV-CAR", "JEEP", "SUV", "TAXI", "FOUR WHEELER"},
    "TRUCK": {"TRUCK", "GOODS", "GOODS CARRIER", "HGV", "HMV", "LGV", "TRAILER", "TANKER"},
    "BUS": {"BUS", "OMNIBUS", "OMNI BUS", "PSV", "STAGE CARRIAGE", "MINIBUS"},
    "BICYCLE": {"BICYCLE", "CYCLE", "NON-MOTORISED"},
}

# Statuses that are a problem regardless of what the vehicle looks like.
FLAGGED_STATUSES = {"STOLEN", "BLACKLISTED", "WANTED", "SUSPENDED", "IMPOUNDED"}


def canonical_class(raw: str | None) -> str | None:
    """Map a registry's wording onto a class the detector can produce."""
    if not raw:
        return None
    text = str(raw).strip().upper()
    for canonical, aliases in CLASS_ALIASES.items():
        if text == canonical or text in aliases:
            return canonical
    return None


@dataclass(frozen=True)
class VehicleRecord:
    """What the registry says is registered against a plate."""

    plate: str
    vehicle_class: str | None = None
    make: str | None = None
    model: str | None = None
    colour: str | None = None
    status: str = "ACTIVE"
    note: str = ""

    @property
    def flagged(self) -> bool:
        return self.status.upper() in FLAGGED_STATUSES

    @property
    def canonical_class(self) -> str | None:
        return canonical_class(self.vehicle_class)

    def to_dict(self) -> dict:
        """Vehicle attributes only - owner details stay in the registry."""
        return {
            "plate": self.plate,
            "vehicle_class": self.vehicle_class,
            "make": self.make,
            "model": self.model,
            "colour": self.colour,
            "status": self.status,
            "flagged": self.flagged,
            "note": self.note,
        }


class RegistryBackend(Protocol):
    """Where registration data comes from.

    A real, authorised RTO/VAHAN client implements this and nothing else in
    IBVAP has to change.
    """

    def lookup(self, plate: str) -> VehicleRecord | None:
        ...


@dataclass
class FileRegistry:
    """A registry extract held as a local JSON file."""

    path: Path
    records: dict[str, VehicleRecord] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"vehicle registry not found: {self.path}")
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for entry in raw.get("vehicles", []):
            plate = str(entry.get("plate", "")).replace(" ", "").upper()
            if not plate:
                continue
            self.records[plate] = VehicleRecord(
                plate=plate,
                vehicle_class=entry.get("vehicle_class"),
                make=entry.get("make"),
                model=entry.get("model"),
                colour=entry.get("colour"),
                status=str(entry.get("status", "ACTIVE")).upper(),
                note=str(entry.get("note", "")),
            )

    def lookup(self, plate: str) -> VehicleRecord | None:
        return self.records.get((plate or "").replace(" ", "").upper())

    def __len__(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class RegistryCheck:
    """The outcome of comparing a plate against what is registered to it."""

    plate: str
    observed_class: str
    record: VehicleRecord | None
    known: bool
    mismatch: bool
    flagged: bool
    detail: str

    def to_dict(self) -> dict:
        return {
            "plate": self.plate,
            "observed_class": self.observed_class,
            "known": self.known,
            "mismatch": self.mismatch,
            "flagged": self.flagged,
            "detail": self.detail,
            "record": self.record.to_dict() if self.record else None,
        }


@dataclass
class VehicleRegistry:
    """Checks a read plate against the registry behind it."""

    backend: RegistryBackend | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "VehicleRegistry":
        return cls(backend=FileRegistry(Path(path)))

    @property
    def enabled(self) -> bool:
        return self.backend is not None

    def check(self, plate: str, observed_class: str) -> RegistryCheck | None:
        """Compare a plate against its registration. None when no registry is loaded."""
        if self.backend is None or not plate:
            return None
        if category_of(observed_class) != VEHICLE:
            return None

        record = self.backend.lookup(plate)
        if record is None:
            return RegistryCheck(
                plate=plate,
                observed_class=observed_class,
                record=None,
                known=False,
                mismatch=False,
                flagged=False,
                detail="plate is not in the registry extract",
            )

        registered = record.canonical_class
        # An unrecognised class in the extract is not a mismatch: it means the
        # registry used wording this build does not know, and asserting a
        # mismatch on that would cry wolf at every unusual vehicle type.
        mismatch = bool(registered and registered != observed_class.upper())

        if record.flagged:
            detail = f"registry status {record.status}"
        elif mismatch:
            detail = (
                f"plate is registered to a {registered.lower()}, "
                f"camera sees a {observed_class.lower()}"
            )
        elif registered is None:
            detail = f"registered class {record.vehicle_class!r} not recognised"
        else:
            detail = f"matches registration ({registered.lower()})"

        return RegistryCheck(
            plate=plate,
            observed_class=observed_class,
            record=record,
            known=True,
            mismatch=mismatch,
            flagged=record.flagged,
            detail=detail,
        )
