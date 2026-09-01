"""Short-term movement history per track.

The Context Engine needs a few seconds of where each person has been in order
to derive speed, heading and dwell. Everything is stored in **normalized**
(0..1) frame coordinates so the derived figures do not change when the camera
resolution or the decode scale changes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

Point = tuple[float, float]


@dataclass(frozen=True)
class TrackPoint:
    """One observation of one track."""

    timestamp: float
    position: Point  # normalized ground point
    zone_name: str | None
    # Apparent size of the person as a fraction of frame height. Cameras are
    # mounted high and look outward, so a large value means the person is very
    # close to the camera itself.
    height_ratio: float = 0.0


@dataclass
class TrackRecord:
    """The rolling window of observations for a single track."""

    track_id: int
    window: int = 24
    points: deque[TrackPoint] = field(default_factory=deque)
    zone_entered_at: float | None = None
    current_zone: str | None = None

    def __post_init__(self) -> None:
        self.points = deque(self.points, maxlen=self.window)

    def observe(self, point: TrackPoint) -> None:
        if point.zone_name != self.current_zone:
            self.current_zone = point.zone_name
            self.zone_entered_at = point.timestamp
        self.points.append(point)

    @property
    def first(self) -> TrackPoint:
        return self.points[0]

    @property
    def last(self) -> TrackPoint:
        return self.points[-1]

    @property
    def age_seconds(self) -> float:
        return max(0.0, self.last.timestamp - self.first.timestamp)

    def dwell_seconds(self, now: float) -> float:
        """How long the track has been continuously inside its current zone."""
        if self.current_zone is None or self.zone_entered_at is None:
            return 0.0
        return max(0.0, now - self.zone_entered_at)

    def displacement(self) -> tuple[float, float]:
        """Vector from the oldest to the newest point in the window."""
        if len(self.points) < 2:
            return (0.0, 0.0)
        (x1, y1), (x2, y2) = self.first.position, self.last.position
        return (x2 - x1, y2 - y1)

    def step_vectors(self) -> list[tuple[float, float]]:
        """Frame-to-frame movement vectors across the window."""
        points = list(self.points)
        return [
            (b.position[0] - a.position[0], b.position[1] - a.position[1])
            for a, b in zip(points, points[1:])
        ]

    def growth_rate(self) -> float:
        """How fast the person is filling more of the frame, per second.

        Positive means they are getting closer to the camera. This is the only
        signal available for movement *along* the camera axis, which a
        ground-plane position cannot see.
        """
        if len(self.points) < 2:
            return 0.0
        elapsed = self.age_seconds
        if elapsed <= 0:
            return 0.0
        return (self.last.height_ratio - self.first.height_ratio) / elapsed

    def heading_variance(self, min_step: float = 1e-4) -> float:
        """How much the direction of travel wanders, from 0.0 to 1.0.

        0.0 is a perfectly straight line, 1.0 is completely inconsistent. It is
        the circular variance of the per-step headings: average the unit
        vectors, and a short resultant means they pointed all over the place.
        Steps too small to have a meaningful direction are ignored.
        """
        vectors = [
            (dx, dy) for dx, dy in self.step_vectors()
            if (dx * dx + dy * dy) ** 0.5 >= min_step
        ]
        if len(vectors) < 2:
            return 0.0
        sum_x = sum_y = 0.0
        for dx, dy in vectors:
            length = (dx * dx + dy * dy) ** 0.5
            sum_x += dx / length
            sum_y += dy / length
        resultant = ((sum_x / len(vectors)) ** 2 + (sum_y / len(vectors)) ** 2) ** 0.5
        return max(0.0, min(1.0, 1.0 - resultant))


@dataclass
class TrackHistory:
    """Holds a `TrackRecord` per active track and forgets stale ones."""

    window: int = 24
    forget_after: float = 60.0
    _records: dict[int, TrackRecord] = field(default_factory=dict, init=False)

    def observe(
        self,
        track_id: int,
        position: Point,
        zone_name: str | None,
        now: float,
        height_ratio: float = 0.0,
    ) -> TrackRecord:
        record = self._records.get(track_id)
        if record is None:
            record = TrackRecord(track_id=track_id, window=self.window)
            self._records[track_id] = record
        record.observe(
            TrackPoint(
                timestamp=now,
                position=position,
                zone_name=zone_name,
                height_ratio=height_ratio,
            )
        )
        self._prune(now)
        return record

    def get(self, track_id: int) -> TrackRecord | None:
        return self._records.get(track_id)

    def forget(self, track_id: int) -> None:
        self._records.pop(track_id, None)

    def _prune(self, now: float) -> None:
        stale = [
            tid for tid, rec in self._records.items()
            if rec.points and now - rec.last.timestamp > self.forget_after
        ]
        for tid in stale:
            del self._records[tid]

    def __len__(self) -> int:
        return len(self._records)

    def reset(self) -> None:
        self._records.clear()
