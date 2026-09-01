"""Pure-Python 2D geometry helpers.

Kept dependency-free on purpose: the zone logic must stay testable without
OpenCV, NumPy or a model download.
"""

from __future__ import annotations

from typing import Iterable, Sequence

Point = tuple[float, float]
Polygon = Sequence[Point]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray-casting point-in-polygon test.

    Points exactly on an edge are treated as inside so a person standing on
    the fence line is never silently ignored.
    """
    if len(polygon) < 3:
        return False

    x, y = point
    if _on_boundary(point, polygon):
        return True

    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Does the edge straddle the horizontal ray from the point?
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1)
            if x < x1 + t * (x2 - x1):
                inside = not inside
    return inside


def _on_boundary(point: Point, polygon: Polygon, eps: float = 1e-9) -> bool:
    x, y = point
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) > eps:
            continue
        if min(x1, x2) - eps <= x <= max(x1, x2) + eps and \
           min(y1, y2) - eps <= y <= max(y1, y2) + eps:
            return True
    return False


def normalize(point: Point, width: int, height: int) -> Point:
    """Pixel coordinates -> resolution-independent 0..1 coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError("frame width and height must be positive")
    return (point[0] / width, point[1] / height)


def denormalize(point: Point, width: int, height: int) -> tuple[int, int]:
    """0..1 coordinates -> pixel coordinates."""
    return (int(round(point[0] * width)), int(round(point[1] * height)))


def denormalize_polygon(polygon: Polygon, width: int, height: int) -> list[tuple[int, int]]:
    return [denormalize(p, width, height) for p in polygon]


def validate_polygon(polygon: Iterable[Sequence[float]]) -> list[Point]:
    """Coerce raw config data into a normalized polygon, or raise."""
    points: list[Point] = []
    for raw in polygon:
        if len(raw) != 2:
            raise ValueError(f"polygon point must have 2 coordinates, got {raw!r}")
        x, y = float(raw[0]), float(raw[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"polygon point {(x, y)} is outside 0..1 - zones use normalized coordinates"
            )
        points.append((x, y))
    if len(points) < 3:
        raise ValueError("a zone polygon needs at least 3 points")
    return points


def point_to_segment_distance(point: Point, a: Point, b: Point) -> float:
    """Shortest distance from `point` to the segment a-b."""
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    # Project the point onto the segment, clamped to its ends.
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def distance_to_polygon(point: Point, polygon: Polygon) -> float:
    """Distance from a point to a polygon, or 0.0 if the point is inside.

    Used to tell whether a track is closing on a fence it has not entered yet.
    """
    if len(polygon) < 3:
        return float("inf")
    if point_in_polygon(point, polygon):
        return 0.0
    n = len(polygon)
    return min(
        point_to_segment_distance(point, polygon[i], polygon[(i + 1) % n]) for i in range(n)
    )


def closest_point_on_segment(point: Point, a: Point, b: Point) -> Point:
    """The point on segment a-b nearest to `point`."""
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return (ax, ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return (ax + t * dx, ay + t * dy)


def closest_point_on_polygon(point: Point, polygon: Polygon) -> Point:
    """Nearest point on a polygon's boundary, or the point itself if inside.

    Needed to measure a real-world distance to a zone: project both this point
    and the original through the ground-plane homography.
    """
    if len(polygon) < 3 or point_in_polygon(point, polygon):
        return point
    n = len(polygon)
    candidates = [
        closest_point_on_segment(point, polygon[i], polygon[(i + 1) % n]) for i in range(n)
    ]
    return min(
        candidates,
        key=lambda c: (c[0] - point[0]) ** 2 + (c[1] - point[1]) ** 2,
    )
