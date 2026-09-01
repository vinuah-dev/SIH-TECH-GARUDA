"""Intrusion state machine and the alert record it produces."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from ..behaviour.engine import Behaviour
from ..context.engine import TrackContext
from ..detection.base import Detection
from ..risk.engine import RiskAssessment
from ..zones.manager import Zone

DEFAULT_ALERT_KINDS = frozenset({"RESTRICTED"})

# Ordering used to decide whether a situation has got worse since the last alert.
SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

# Camera tampering is not tied to a zone - someone reaching for the lens is a
# threat wherever they are standing - so it gets its own trigger key.
TAMPER_KEY = "__CAMERA__"


@dataclass(frozen=True)
class IntrusionEvent:
    """One confirmed security event, ready to log or display.

    Named for the first case it served, but now broader: `zone` is None for a
    threat to the camera itself, and `detection` is None for an event with no
    person in it at all - a camera whose link has been cut has no subject to
    detect, which is precisely what makes it worth reporting.
    """

    event_id: str
    timestamp: datetime
    camera_id: str
    event_type: str
    detection: Detection | None
    zone: Zone | None
    risk: RiskAssessment
    frame_index: int
    evidence_path: str | None = None
    clip_path: str | None = None
    plate_path: str | None = None
    plate: dict | None = None
    registry: dict | None = None
    identity: dict | None = None
    subject: dict | None = None
    track_context: TrackContext | None = None
    behaviours: list[Behaviour] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(timespec="seconds"),
            "camera_id": self.camera_id,
            "event_type": self.event_type,
            "frame_index": self.frame_index,
            "object": None if self.detection is None else {
                "label": self.detection.label,
                "confidence": round(self.detection.confidence, 4),
                "bbox": list(self.detection.bbox),
                "ground_point": list(self.detection.ground_point),
                "track_id": self.detection.track_id,
                "track_label": self.detection.track_label,
            },
            "zone": None if self.zone is None else {
                "name": self.zone.name,
                "kind": self.zone.kind,
                "base_risk": self.zone.base_risk,
            },
            "risk": {
                "score": self.risk.score,
                "severity": self.risk.severity,
                "reason": self.risk.reason,
                "factors": [
                    {"label": f.label, "points": f.points, "detail": f.detail}
                    for f in self.risk.factors
                ],
            },
            "context": self.track_context.to_dict() if self.track_context else None,
            "behaviours": [
                {"name": b.name, "points": b.points, "detail": b.detail} for b in self.behaviours
            ],
            "evidence_path": self.evidence_path,
            "clip_path": self.clip_path,
            "plate_path": self.plate_path,
            "plate": self.plate,
            "registry": self.registry,
            "identity": self.identity,
            "subject": self.subject,
        }


@dataclass
class _TrackState:
    zone_name: str | None = None
    last_seen: float = 0.0
    # Consecutive confirming frames per trigger. A person can be inside a zone
    # *and* reaching for the camera; those are independent alerts.
    frames: dict[str, int] = field(default_factory=dict)
    last_alert_at: dict[str, float] = field(default_factory=dict)
    last_severity: dict[str, str] = field(default_factory=dict)


@dataclass
class IntrusionMonitor:
    """Decides *when* a zone occupancy becomes an alert.

    Three rules keep the console usable without hiding a worsening situation:
      * `confirm_frames` - a box must sit inside the zone for N consecutive
        frames, which suppresses single-frame detector flicker on the boundary.
      * `cooldown_seconds` - the same track cannot re-alert on the same zone
        until the cooldown expires (first-pass alert-fatigue reduction).
      * escalation - a cooldown never suppresses an alert whose severity is
        *higher* than the one already raised for that track and zone. An
        intruder who stops and loiters must reach the operator.

    All times are footage seconds (`Frame.stream_time`), not wall clock.
    """

    confirm_frames: int = 3
    cooldown_seconds: float = 30.0
    alert_kinds: frozenset[str] = DEFAULT_ALERT_KINDS
    forget_after: float = 60.0

    _tracks: dict[int, _TrackState] = field(default_factory=dict, init=False)

    def update(
        self,
        track_id: int | None,
        zone: Zone | None,
        now: float,
        severity: str | None = None,
    ) -> bool:
        """Feed one frame of zone occupancy. True when a fence alert should fire."""
        if track_id is None:
            return False

        state = self._touch(track_id, now)

        zone_name = zone.name if zone else None
        if zone_name != state.zone_name:
            if state.zone_name:
                state.frames.pop(state.zone_name, None)
            state.zone_name = zone_name

        if zone is None:
            return False

        state.frames[zone.name] = state.frames.get(zone.name, 0) + 1
        if zone.kind not in self.alert_kinds:
            return False
        return self._should_fire(state, zone.name, now, severity)

    def update_tampering(
        self,
        track_id: int | None,
        now: float,
        tampering: bool,
        severity: str | None = None,
    ) -> bool:
        """Feed one frame of camera-tampering state. True to raise an alert.

        Deliberately independent of any zone: a person reaching for the lens is
        a threat wherever they are standing, and blinding the camera defeats
        every zone rule anyway.
        """
        if track_id is None:
            return False

        state = self._touch(track_id, now)
        if not tampering:
            state.frames.pop(TAMPER_KEY, None)
            return False

        state.frames[TAMPER_KEY] = state.frames.get(TAMPER_KEY, 0) + 1
        return self._should_fire(state, TAMPER_KEY, now, severity)

    # ---------------------------------------------------------------- shared

    def _touch(self, track_id: int, now: float) -> _TrackState:
        self._prune(now)
        state = self._tracks.setdefault(track_id, _TrackState())
        state.last_seen = now
        return state

    def _should_fire(
        self, state: _TrackState, key: str, now: float, severity: str | None
    ) -> bool:
        if state.frames.get(key, 0) < self.confirm_frames:
            return False

        # Past the confirmation threshold the cooldown is what stops the same
        # standing person from re-alerting on every subsequent frame - unless
        # the situation has escalated to a higher severity than last reported.
        last = state.last_alert_at.get(key)
        if last is not None and now - last < self.cooldown_seconds:
            if not self._escalated(state, key, severity):
                return False

        state.last_alert_at[key] = now
        if severity:
            state.last_severity[key] = severity
        return True

    @staticmethod
    def _escalated(state: _TrackState, zone_name: str, severity: str | None) -> bool:
        if severity is None:
            return False
        previous = state.last_severity.get(zone_name)
        if previous is None:
            return False
        return SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(previous, 0)

    def _prune(self, now: float) -> None:
        stale = [tid for tid, st in self._tracks.items() if now - st.last_seen > self.forget_after]
        for tid in stale:
            del self._tracks[tid]

    def reset(self) -> None:
        self._tracks.clear()


def new_event_id() -> str:
    return uuid.uuid4().hex[:12].upper()
