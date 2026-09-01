"""Frame annotation, shared by the live view window and evidence snapshots."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

import cv2
import numpy as np

from ..detection.base import Detection
from ..zones.manager import Zone

# BGR
ZONE_COLORS = {
    "RESTRICTED": (60, 60, 220),
    "WATCH": (40, 180, 220),
    "PATROL": (140, 180, 140),
}
BOX_CLEAR = (120, 220, 120)
BOX_ALERT = (60, 60, 220)
HUD_BG = (24, 24, 24)
HUD_FG = (235, 235, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_zones(frame: np.ndarray, zones: Iterable[Zone], alpha: float = 0.22) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    for zone in zones:
        color = ZONE_COLORS.get(zone.kind, (200, 200, 200))
        pts = np.array(zone.pixels(width, height), dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        anchor = tuple(pts[pts[:, 1].argmin()])
        cv2.putText(frame, zone.name, (anchor[0] + 6, anchor[1] + 20), FONT, 0.5, color, 1,
                    cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    zone: Zone | None,
    alerting: bool = False,
) -> np.ndarray:
    x1, y1, x2, y2 = detection.bbox
    color = BOX_ALERT if alerting else BOX_CLEAR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{detection.label} {detection.track_label} {detection.confidence:.0%}"
    if zone:
        label += f" | {zone.name}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, max(10, y1 - 5)), FONT, 0.5, (15, 15, 15), 1, cv2.LINE_AA)

    # The point actually tested against the virtual fence.
    gx, gy = detection.ground_point
    cv2.circle(frame, (gx, gy), 4, color, -1)
    cv2.circle(frame, (gx, gy), 7, color, 1)
    return frame


def draw_hud(
    frame: np.ndarray,
    camera_id: str,
    frame_index: int,
    alerts: int,
    fps: float,
    when: datetime | None = None,
) -> np.ndarray:
    when = when or datetime.now()
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, 30), HUD_BG, -1)
    left = f"IBVAP | {camera_id} | frame {frame_index} | {fps:.1f} fps"
    right = f"{when:%Y-%m-%d %H:%M:%S} | ALERTS {alerts}"
    cv2.putText(frame, left, (10, 20), FONT, 0.5, HUD_FG, 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(right, FONT, 0.5, 1)
    cv2.putText(frame, right, (width - tw - 10, 20), FONT, 0.5, HUD_FG, 1, cv2.LINE_AA)
    return frame


def annotate(
    frame: np.ndarray,
    zones: Iterable[Zone],
    detections: Sequence[tuple[Detection, Zone | None, bool]],
    camera_id: str,
    frame_index: int,
    alerts: int,
    fps: float,
) -> np.ndarray:
    """Full annotated copy of a frame: zones, boxes, HUD."""
    canvas = frame.copy()
    draw_zones(canvas, zones)
    for detection, zone, alerting in detections:
        draw_detection(canvas, detection, zone, alerting)
    draw_hud(canvas, camera_id, frame_index, alerts, fps)
    return canvas
