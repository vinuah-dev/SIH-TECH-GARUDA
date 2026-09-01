"""Operator-facing console rendering for detections and alerts."""

from __future__ import annotations

from ..behaviour.engine import Behaviour
from ..context.engine import TrackContext
from ..detection.base import Detection
from ..ui import WIDTH, c, field, severity_color
from ..zones.manager import Zone
from .events import IntrusionEvent

_SEP = "=" * WIDTH


def render_detection(
    detection: Detection,
    zone: Zone | None,
    camera_id: str,
    frame_index: int,
    track: TrackContext | None = None,
    behaviours: list[Behaviour] | None = None,
) -> str:
    """Compact one-line detection record."""
    zone_name = zone.name if zone else "OPEN"
    zone_style = "yellow" if zone and zone.kind == "RESTRICTED" else "white"
    x, y = detection.ground_point
    line = (
        f"{c('[DETECTION]', 'bold')} "
        f"{detection.label:<6} {c(detection.track_label, 'cyan')}  "
        f"conf {detection.confidence:.0%}  "
        f"zone {c(zone_name, zone_style)}  "
        f"ground ({x},{y})  cam {camera_id}  frame {frame_index}"
    )
    if track is not None:
        line += f"  {c(render_context(track), 'dim')}"
    if behaviours:
        tags = " ".join(f"[{b.name}]" for b in behaviours)
        line += f"  {c(tags, 'yellow')}"
    return line


def render_context(track: TrackContext) -> str:
    """Movement context as a compact suffix: speed, heading, dwell, approach."""
    parts = [f"spd {track.speed_label}", f"hdg {track.heading}"]
    if track.dwell_seconds >= 1:
        parts.append(f"dwell {track.dwell_seconds:.0f}s")
    if track.height_ratio >= 0.35:
        parts.append(f"size {track.height_ratio:.0%}")
    if track.approaching and track.target_zone:
        distance = track.distance_label
        parts.append(f"-> {track.target_zone}" + (f" ({distance})" if distance else ""))
    elif track.distance_label and (track.distance_to_target or 0) > 0:
        parts.append(f"dist {track.distance_label}")
    return " | ".join(parts)


def render_alert(event: IntrusionEvent) -> str:
    """Full security-alert banner."""
    style = severity_color(event.risk.severity)
    lines: list[str] = [
        "",
        c(_SEP, style),
        c("SECURITY ALERT".center(WIDTH), "bold", style),
        c(_SEP, style),
        "",
        field("Type", event.event_type),
        field("Object", event.detection.label if event.detection else "CAMERA"),
        field("Camera", event.camera_id),
        field("Track ID", event.detection.track_label if event.detection else "n/a"),
        field(
            "Confidence",
            f"{event.detection.confidence:.0%}" if event.detection else "n/a",
        ),
        field(
            "Zone",
            f"{event.zone.name} ({event.zone.kind})" if event.zone else "n/a - camera itself",
        ),
        field(
            "Behaviours",
            ", ".join(b.name for b in event.behaviours) if event.behaviours else "none",
        ),
        *([field("Plate", c(event.plate["display"], "bold"))] if event.plate else []),
        *([field("Registered", _registry_line(event.registry))] if event.registry else []),
        field("Severity", c(event.risk.severity, "bold", style)),
        field("Risk Score", c(f"{event.risk.score}/100", "bold", style)),
        field("Time", event.timestamp.strftime("%Y-%m-%d %H:%M:%S")),
        field("Frame", str(event.frame_index)),
        field("Event ID", event.event_id),
        "",
        c("Reason:", "bold"),
        event.risk.reason,
        "",
        c("Contributing factors:", "bold"),
    ]
    lines += [f"  {factor}" for factor in event.risk.factors]
    if event.track_context is not None:
        lines += ["", c("Movement context:", "bold"), f"  {render_context(event.track_context)}"]
    if event.evidence_path:
        lines += ["", field("Evidence", event.evidence_path)]
    if event.clip_path:
        lines += [field("Clip", event.clip_path)]
    if event.plate_path:
        lines += [field("Plate img", event.plate_path)]
    lines += ["", c(_SEP, style), ""]
    return "\n".join(lines)


def _registry_line(registry: dict) -> str:
    """What the registry says about this plate, in one line."""
    if not registry.get("known"):
        return "not in registry extract"
    record = registry.get("record") or {}
    described = " ".join(
        str(part) for part in (record.get("colour"), record.get("make"), record.get("model"))
        if part
    )
    label = f"{record.get('vehicle_class', '?')}"
    if described:
        label += f" - {described}"
    if registry.get("flagged"):
        return f"{label}  [{record.get('status')}]"
    if registry.get("mismatch"):
        return f"{label}  [MISMATCH]"
    return label


def render_status(frame_index: int, fps: float, tracks: int, alerts: int) -> str:
    return (
        f"{c('[STATUS]', 'dim')} frame {frame_index:<6} "
        f"fps {fps:5.1f}  active tracks {tracks}  alerts {alerts}"
    )
