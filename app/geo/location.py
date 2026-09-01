"""Where each camera physically is, so alerts can be put on a map.

An alert that says CAM-05 means nothing to somebody who does not already know
where CAM-05 is. On a map it means *this stretch of the river*, and a commander
can send the nearest patrol instead of asking which camera that was.

Two things in here are easy to confuse, and keeping them apart matters:

**A zone is not a geofence.** The virtual fences the pipeline already has live
in *frame* coordinates - fractions of the image, drawn by eye on what the
camera sees. They cannot be drawn on a map, because turning image coordinates
into ground coordinates needs a surveyed homography per camera, which most
posts will not have. A geofence here is the opposite: a polygon in latitude and
longitude, drawn on the map, that knows nothing about any camera's view. Both
are useful; neither converts into the other for free.

**A camera's bearing is not its coverage.** Bearing plus field of view draws a
wedge on a map, and that wedge is a *claim about where the camera points*, not
a measurement of what it can see. A ridge, a wall or a truck blocks most of it.
The wedge is worth drawing so long as nobody reads it as proven coverage.

Everything here is optional. A camera with no location runs exactly as before,
which is the normal case today - locations get filled in when somebody walks
the post with a GPS.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# Mean radius, in metres. Good to about 0.3% for distances across one post,
# which is far better than the GPS fix that produced the coordinates.
EARTH_RADIUS = 6_371_008.8


class InvalidLocation(ValueError):
    """A coordinate that is not on Earth, caught at load rather than on a map."""


@dataclass(frozen=True)
class CameraLocation:
    """One camera's place on the ground."""

    latitude: float
    longitude: float
    # Compass degrees the camera faces: 0 north, 90 east. None means unsurveyed,
    # which is honest - a wedge drawn from a guessed bearing is worse than none.
    bearing: float | None = None
    # Horizontal field of view in degrees. Together with bearing this draws the
    # wedge; alone it draws nothing.
    field_of_view: float | None = None
    # Height above ground in metres, which decides how far the camera can
    # usefully see and whether somebody can reach it.
    height_m: float | None = None
    site: str = ""          # "Gate A", "River Bank" - what an operator calls it

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise InvalidLocation(f"latitude {self.latitude} is not between -90 and 90")
        if not -180.0 <= self.longitude <= 180.0:
            raise InvalidLocation(
                f"longitude {self.longitude} is not between -180 and 180"
            )
        if self.bearing is not None and not 0.0 <= self.bearing < 360.0:
            raise InvalidLocation(f"bearing {self.bearing} is not between 0 and 360")
        if self.field_of_view is not None and not 0.0 < self.field_of_view <= 360.0:
            raise InvalidLocation(
                f"field of view {self.field_of_view} is not between 0 and 360"
            )

    @classmethod
    def from_dict(cls, raw: dict | None) -> "CameraLocation | None":
        """Read a location from a camera entry, or None if it has none."""
        if not raw:
            return None
        try:
            latitude = float(raw["lat"] if "lat" in raw else raw["latitude"])
            longitude = float(raw["lon"] if "lon" in raw else raw["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidLocation(
                f"a camera location needs numeric lat and lon: {raw!r}"
            ) from exc

        def optional(*names):
            for name in names:
                if raw.get(name) is not None:
                    return float(raw[name])
            return None

        return cls(
            latitude=latitude,
            longitude=longitude,
            bearing=optional("bearing", "heading"),
            field_of_view=optional("fov", "field_of_view"),
            height_m=optional("height_m", "height"),
            site=str(raw.get("site") or raw.get("name") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "lat": round(self.latitude, 7),
            "lon": round(self.longitude, 7),
            "bearing": self.bearing,
            "fov": self.field_of_view,
            "height_m": self.height_m,
            "site": self.site,
        }

    # ------------------------------------------------------------- geometry

    def distance_to(self, other: "CameraLocation") -> float:
        """Metres between two cameras, along the surface.

        Haversine rather than flat trigonometry: over one post the difference
        is centimetres, but a post on a border may be surveyed in one file with
        another a hundred kilometres away, where flat maths is simply wrong.
        """
        lat1, lat2 = math.radians(self.latitude), math.radians(other.latitude)
        dlat = lat2 - lat1
        dlon = math.radians(other.longitude - self.longitude)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        return 2 * EARTH_RADIUS * math.asin(min(1.0, math.sqrt(a)))

    def coverage_wedge(self, reach_m: float = 120.0, points: int = 12) -> list[list[float]]:
        """The area the camera *points at*, as a polygon for a map.

        Emphatically not measured coverage - a ridge or a parked truck takes
        most of it away. It is drawn so an operator can see which way a camera
        faces, and it is empty unless both bearing and field of view are known,
        because a wedge from a guessed bearing points confidently at the wrong
        hillside.
        """
        if self.bearing is None or self.field_of_view is None:
            return []

        half = self.field_of_view / 2.0
        edge = [[self.latitude, self.longitude]]
        for i in range(points + 1):
            angle = self.bearing - half + (self.field_of_view * i / points)
            edge.append(list(self._offset(angle, reach_m)))
        edge.append([self.latitude, self.longitude])
        return edge

    def _offset(self, bearing_deg: float, metres: float) -> tuple[float, float]:
        """The point `metres` away on this compass bearing."""
        angular = metres / EARTH_RADIUS
        bearing = math.radians(bearing_deg)
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)

        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular)
            + math.cos(lat1) * math.sin(angular) * math.cos(bearing)
        )
        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(angular) * math.cos(lat1),
            math.cos(angular) - math.sin(lat1) * math.sin(lat2),
        )
        return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


@dataclass(frozen=True)
class Geofence:
    """A named area on the map, in latitude and longitude.

    Nothing in the running pipeline tests against these yet - a detection is
    located in a frame, not on the ground, and bridging that needs a surveyed
    homography the posts do not have. They exist so the map has areas to draw
    and so cameras can be grouped by the sector they sit in.
    """

    name: str
    polygon: tuple[tuple[float, float], ...]
    kind: str = "SECTOR"        # SECTOR | RESTRICTED | PATROL | BORDER
    description: str = ""

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise InvalidLocation(
                f"geofence {self.name!r} needs at least 3 points, got {len(self.polygon)}"
            )
        for lat, lon in self.polygon:
            if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
                raise InvalidLocation(
                    f"geofence {self.name!r} has a point off the Earth: {(lat, lon)}"
                )

    @classmethod
    def from_dict(cls, raw: dict) -> "Geofence":
        points = raw.get("polygon") or raw.get("points") or []
        return cls(
            name=str(raw.get("name") or "AREA"),
            polygon=tuple((float(p[0]), float(p[1])) for p in points),
            kind=str(raw.get("kind") or "SECTOR").upper(),
            description=str(raw.get("description") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "polygon": [[lat, lon] for lat, lon in self.polygon],
        }

    def contains(self, latitude: float, longitude: float) -> bool:
        """Ray casting, the same method the frame zones use, on ground coordinates.

        Longitude is taken as given rather than unwrapped, so a fence spanning
        the 180th meridian would be wrong. No land border this is built for goes
        near it, and pretending to handle a case that cannot be tested here
        would be worse than saying so.
        """
        inside = False
        count = len(self.polygon)
        for i in range(count):
            lat1, lon1 = self.polygon[i]
            lat2, lon2 = self.polygon[(i + 1) % count]
            if (lat1 > latitude) != (lat2 > latitude):
                crossing = lon1 + (latitude - lat1) * (lon2 - lon1) / (lat2 - lat1)
                if longitude < crossing:
                    inside = not inside
        return inside


@dataclass
class SiteMap:
    """Everything the map knows: where the cameras are, and what areas exist."""

    cameras: dict[str, CameraLocation] = field(default_factory=dict)
    geofences: tuple[Geofence, ...] = ()

    @classmethod
    def from_file(cls, path: str | Path) -> "SiteMap":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"site map not found: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, raw: dict) -> "SiteMap":
        cameras = {}
        for camera_id, entry in (raw.get("cameras") or {}).items():
            located = CameraLocation.from_dict(entry)
            if located is not None:
                cameras[str(camera_id)] = located
        return cls(
            cameras=cameras,
            geofences=tuple(Geofence.from_dict(g) for g in (raw.get("geofences") or [])),
        )

    def locate(self, camera_id: str) -> CameraLocation | None:
        return self.cameras.get(camera_id)

    def fences_around(self, camera_id: str) -> list[str]:
        """Which named areas this camera stands inside."""
        where = self.locate(camera_id)
        if where is None:
            return []
        return [
            fence.name for fence in self.geofences
            if fence.contains(where.latitude, where.longitude)
        ]

    def nearest(self, camera_id: str, limit: int = 3) -> list[tuple[str, float]]:
        """The closest other cameras and how far away they are, in metres.

        For a map that has to answer "a person left CAM-02 heading east - which
        camera should be watched next?"
        """
        here = self.locate(camera_id)
        if here is None:
            return []
        others = [
            (other_id, here.distance_to(other))
            for other_id, other in self.cameras.items()
            if other_id != camera_id
        ]
        return sorted(others, key=lambda pair: pair[1])[:limit]

    def to_dict(self) -> dict:
        """Everything a map view needs, in one payload."""
        return {
            "cameras": [
                {
                    "camera_id": camera_id,
                    **location.to_dict(),
                    "coverage": location.coverage_wedge(),
                    "geofences": self.fences_around(camera_id),
                }
                for camera_id, location in sorted(self.cameras.items())
            ],
            "geofences": [fence.to_dict() for fence in self.geofences],
            "located": len(self.cameras),
        }

    def __len__(self) -> int:
        return len(self.cameras)
