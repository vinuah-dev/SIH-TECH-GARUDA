"""Context Engine.

Turns raw per-frame detections plus short-term track history into the
situational facts the Risk Engine reasons about:

    time + location + zone + direction + speed + duration

It classifies nothing. Deciding that a dwell of 40 s *is* loitering belongs to
the Behaviour Engine; this stage only measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..calibration import GroundPlane
from ..zones.geometry import Point, closest_point_on_polygon, distance_to_polygon
from ..zones.manager import Zone, ZoneManager
from .history import TrackHistory, TrackRecord
from .scene import SceneContext

# Below this speed (normalized frame widths per second) heading is meaningless
# noise, so direction-based signals are suppressed.
STATIONARY_SPEED = 0.01

# The same idea once the camera is calibrated, in metres per second. A person
# shifting their weight is not "moving"; a slow walk is about 1.0 m/s.
STATIONARY_SPEED_MPS = 0.25

# A track must close this much distance (normalized) across its window before
# it counts as approaching, which keeps detector jitter from reading as intent.
APPROACH_EPSILON = 0.01

COMPASS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


@dataclass(frozen=True)
class TrackContext:
    """Everything the Risk Engine knows about one person at one instant."""

    track_id: int
    position: Point
    zone: Zone | None
    scene: SceneContext

    speed: float = 0.0            # normalized frame widths per second
    heading: str = "-"            # compass label, "-" when effectively stationary
    dwell_seconds: float = 0.0    # continuous time in the current zone
    track_age: float = 0.0        # total time this track has been observed
    distance_to_target: float | None = None   # normalized distance to nearest armed zone
    target_zone: str | None = None
    approaching: bool = False     # closing on that armed zone
    heading_variance: float = 0.0  # 0.0 straight line .. 1.0 completely erratic
    samples: int = 1

    # How large the person appears. Cameras look outward from a high mount, so
    # a person filling the frame is physically at the camera.
    height_ratio: float = 0.0      # bbox height / frame height
    coverage: float = 0.0          # bbox area / frame area
    camera_growth: float = 0.0     # height_ratio gained per second (+ = closing in)
    bottom_clipped: bool = False   # feet below the frame edge - very close, or occluded

    # Populated only when the camera has a ground-plane calibration.
    speed_mps: float | None = None
    distance_to_target_m: float | None = None
    unit: str = "m"

    @property
    def calibrated(self) -> bool:
        return self.speed_mps is not None

    @property
    def is_moving(self) -> bool:
        """Movement threshold in real units when calibrated, frame units otherwise."""
        if self.speed_mps is not None:
            return self.speed_mps >= STATIONARY_SPEED_MPS
        return self.speed >= STATIONARY_SPEED

    @property
    def inside_zone(self) -> bool:
        return self.zone is not None

    @property
    def near_camera(self) -> bool:
        """True when the person is close enough to physically reach the camera."""
        return self.height_ratio >= 0.55

    @property
    def speed_label(self) -> str:
        """Speed phrased in whichever units this camera can actually justify."""
        if self.speed_mps is not None:
            return f"{self.speed_mps:.2f} {self.unit}/s"
        return f"{self.speed:.2f} w/s"

    @property
    def distance_label(self) -> str | None:
        if self.distance_to_target_m is not None:
            return f"{self.distance_to_target_m:.1f} {self.unit}"
        if self.distance_to_target is not None:
            return f"{self.distance_to_target:.2f} w"
        return None

    def to_dict(self) -> dict:
        return {
            "speed": round(self.speed, 4),
            "heading": self.heading,
            "dwell_seconds": round(self.dwell_seconds, 2),
            "track_age_seconds": round(self.track_age, 2),
            "distance_to_target": (
                None if self.distance_to_target is None else round(self.distance_to_target, 4)
            ),
            "target_zone": self.target_zone,
            "approaching": self.approaching,
            "heading_variance": round(self.heading_variance, 3),
            "samples": self.samples,
            "height_ratio": round(self.height_ratio, 3),
            "coverage": round(self.coverage, 4),
            "camera_growth": round(self.camera_growth, 4),
            "bottom_clipped": self.bottom_clipped,
            "calibrated": self.calibrated,
            "speed_mps": None if self.speed_mps is None else round(self.speed_mps, 3),
            "distance_to_target_m": (
                None if self.distance_to_target_m is None else round(self.distance_to_target_m, 2)
            ),
        }


@dataclass
class ContextEngine:
    """Derives movement context for each tracked person, frame by frame."""

    zones: ZoneManager
    armed_kinds: frozenset[str] = frozenset({"RESTRICTED"})
    window: int = 24
    ground: GroundPlane | None = None
    history: TrackHistory = field(init=False)

    def __post_init__(self) -> None:
        self.history = TrackHistory(window=self.window)
        self._armed = [z for z in self.zones if z.kind in self.armed_kinds]
        # The calibration travels with the camera config unless overridden.
        if self.ground is None:
            self.ground = self.zones.ground

    def observe(
        self,
        track_id: int,
        position: Point,
        zone: Zone | None,
        scene: SceneContext,
        now: float,
        bbox_norm: tuple[float, float, float, float] | None = None,
    ) -> TrackContext:
        height_ratio, coverage, clipped = self._apparent_size(bbox_norm)
        record = self.history.observe(
            track_id, position, zone.name if zone else None, now, height_ratio=height_ratio
        )
        speed, heading = self._movement(record)
        target, distance = self._nearest_armed_zone(position)
        approaching = self._is_approaching(record, target) if target else False
        speed_mps = self._metric_speed(record)
        distance_m = self._metric_distance(position, target)

        return TrackContext(
            track_id=track_id,
            position=position,
            zone=zone,
            scene=scene,
            speed=speed,
            heading=heading,
            dwell_seconds=record.dwell_seconds(now),
            track_age=record.age_seconds,
            distance_to_target=distance,
            target_zone=target.name if target else None,
            approaching=approaching,
            heading_variance=record.heading_variance(),
            samples=len(record.points),
            height_ratio=height_ratio,
            coverage=coverage,
            camera_growth=record.growth_rate(),
            bottom_clipped=clipped,
            speed_mps=speed_mps,
            distance_to_target_m=distance_m,
            unit=self.ground.unit if self.ground else "m",
        )

    @staticmethod
    def _apparent_size(
        bbox_norm: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, bool]:
        """Height fraction, area fraction, and whether the box is cut off below."""
        if bbox_norm is None:
            return 0.0, 0.0, False
        x1, y1, x2, y2 = bbox_norm
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        # A box running past the bottom edge means the feet are out of shot,
        # which is what happens when someone stands right under the camera.
        return height, width * height, y2 >= 0.995

    def _metric_speed(self, record: TrackRecord) -> float | None:
        """Speed in real units per second, or None on an uncalibrated camera.

        A calibrated camera reports 0.0 rather than None before there is enough
        history, so the units it reports never flip mid-session.
        """
        if self.ground is None:
            return None
        if len(record.points) < 2:
            return 0.0
        elapsed = record.age_seconds
        if elapsed <= 0:
            return 0.0
        travelled = self.ground.distance(record.first.position, record.last.position)
        return travelled / elapsed

    def _metric_distance(self, position: Point, target: Zone | None) -> float | None:
        """Real-world distance to the nearest armed zone, 0.0 when inside it."""
        if self.ground is None or target is None:
            return None
        nearest = closest_point_on_polygon(position, target.polygon)
        return self.ground.distance(position, nearest)

    # ------------------------------------------------------------- internals

    @staticmethod
    def _movement(record: TrackRecord) -> tuple[float, str]:
        if len(record.points) < 2:
            return 0.0, "-"
        elapsed = record.age_seconds
        if elapsed <= 0:
            return 0.0, "-"

        dx, dy = record.displacement()
        speed = math.hypot(dx, dy) / elapsed
        if speed < STATIONARY_SPEED:
            return speed, "-"
        return speed, ContextEngine._compass(dx, dy)

    @staticmethod
    def _compass(dx: float, dy: float) -> str:
        # Image y grows downward, so negate it to read as a normal heading.
        angle = math.degrees(math.atan2(-dy, dx)) % 360
        return COMPASS[int((angle + 22.5) % 360 // 45)]

    def _nearest_armed_zone(self, position: Point) -> tuple[Zone | None, float | None]:
        if not self._armed:
            return None, None
        distances = [(zone, distance_to_polygon(position, zone.polygon)) for zone in self._armed]
        return min(distances, key=lambda pair: pair[1])

    def _is_approaching(self, record: TrackRecord, target: Zone) -> bool:
        """True when the window's oldest point was measurably further away."""
        if len(record.points) < 2:
            return False
        was = distance_to_polygon(record.first.position, target.polygon)
        now = distance_to_polygon(record.last.position, target.polygon)
        return (was - now) > APPROACH_EPSILON

    def forget(self, track_id: int) -> None:
        self.history.forget(track_id)
