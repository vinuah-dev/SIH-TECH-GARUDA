"""Virtual fence definitions and lookup."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..calibration import GroundPlane
from .geometry import Point, denormalize_polygon, point_in_polygon, validate_polygon

# Higher weight wins when a point falls inside overlapping zones.
ZONE_KINDS = {
    "RESTRICTED": 3,
    "WATCH": 2,
    "PATROL": 1,
}


@dataclass(frozen=True)
class Zone:
    """A virtual fence, stored in normalized (0..1) frame coordinates.

    Normalized coordinates mean one zone file keeps working when the camera
    resolution or the decode scale changes.
    """

    name: str
    kind: str
    polygon: list[Point]
    base_risk: int = 50
    description: str = ""
    # Which object categories this fence applies to. A vehicle gate is a fence
    # for people but not for the trucks it exists to admit, and vice versa.
    objects: tuple[str, ...] = ("PERSON", "VEHICLE")

    @property
    def priority(self) -> int:
        return ZONE_KINDS.get(self.kind, 0)

    def contains(self, point_norm: Point) -> bool:
        return point_in_polygon(point_norm, self.polygon)

    def applies_to(self, label: str) -> bool:
        from ..detection.classes import category_of

        return category_of(label) in self.objects

    def pixels(self, width: int, height: int) -> list[tuple[int, int]]:
        return denormalize_polygon(self.polygon, width, height)


@dataclass
class ZoneManager:
    """Holds the per-camera configuration: virtual fences and, optionally, the
    ground-plane calibration that turns image coordinates into metres."""

    zones: list[Zone] = field(default_factory=list)
    ground: GroundPlane | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "ZoneManager":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"zone configuration not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "ZoneManager":
        zones: list[Zone] = []
        for entry in raw.get("zones", []):
            kind = str(entry.get("kind", "WATCH")).upper()
            if kind not in ZONE_KINDS:
                raise ValueError(
                    f"unknown zone kind {kind!r}; expected one of {sorted(ZONE_KINDS)}"
                )
            zones.append(
                Zone(
                    name=str(entry["name"]),
                    kind=kind,
                    polygon=validate_polygon(entry["polygon"]),
                    base_risk=int(entry.get("base_risk", 50)),
                    description=str(entry.get("description", "")),
                    objects=tuple(
                        str(o).upper() for o in entry.get("objects", ["PERSON", "VEHICLE"])
                    ),
                )
            )
        if not zones:
            raise ValueError("zone configuration contains no zones")

        calibration = raw.get("calibration")
        ground = GroundPlane.from_config(calibration) if calibration else None
        return cls(zones=zones, ground=ground)

    def locate(self, point_norm: Point) -> Zone | None:
        """Return the highest-priority zone containing the point, if any."""
        matches = [z for z in self.zones if z.contains(point_norm)]
        if not matches:
            return None
        return max(matches, key=lambda z: (z.priority, z.base_risk))

    def __len__(self) -> int:
        return len(self.zones)

    def __iter__(self):
        return iter(self.zones)
