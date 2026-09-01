"""Explainable rule-based risk scoring.

    context + behaviour + object  ->  score 0-100  ->  explainable alert

Every point in the final score is attributable to a named factor, so an
operator can always see *why* an alert fired. This stage scores; it does not
classify. Behaviours arrive already named from the Behaviour Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..behaviour.engine import Behaviour
from ..context.engine import TrackContext
from ..context.scene import SceneContext
from ..detection.base import Detection
from ..zones.manager import Zone

CRITICAL, HIGH, MEDIUM, LOW = "CRITICAL", "HIGH", "MEDIUM", "LOW"


@dataclass(frozen=True)
class RiskFactor:
    label: str
    points: int
    detail: str

    def __str__(self) -> str:
        return f"{self.points:+d}  {self.label} - {self.detail}"


@dataclass(frozen=True)
class RiskAssessment:
    score: int
    severity: str
    reason: str
    factors: list[RiskFactor] = field(default_factory=list)


@dataclass
class RiskEngine:
    """Turns (object + zone + context) into a 0-100 score with reasons."""

    night_bonus: int = 6
    confidence_pivot: float = 0.60
    confidence_gain: float = 50.0
    confidence_cap: int = 10
    thresholds: tuple[int, int, int] = (90, 65, 40)  # CRITICAL, HIGH, MEDIUM

    # Base risk for an alert that is not tied to a virtual fence, such as
    # someone interfering with the camera itself. Tuned so that tampering at
    # night reaches CRITICAL on its own (45 + 10 confidence + 6 night + 30
    # tampering = 91) while daytime tampering lands at HIGH - in daylight it
    # might still be a maintenance crew.
    zoneless_base_risk: int = 45
    zoneless_label: str = "CAMERA INTEGRITY"

    def assess(
        self,
        detection: Detection | None,
        zone: Zone | None,
        context: SceneContext,
        track: TrackContext | None = None,
        behaviours: list[Behaviour] | None = None,
    ) -> RiskAssessment:
        """Score one detection.

        `track` and `behaviours` are optional so the engine still works before
        any movement history exists - the first frame, or a still image.
        """
        if zone is not None:
            from ..detection.classes import risk_weight

            weight = risk_weight(detection.label) if detection is not None else 1.0
            detail = zone.description or f"presence inside {zone.name}"
            if detection is not None and weight != 1.0:
                detail += f" ({detection.label.lower()}, weight x{weight:g})"
            factors: list[RiskFactor] = [
                RiskFactor(
                    label=f"{zone.kind} ZONE",
                    points=int(round(zone.base_risk * weight)),
                    detail=detail,
                )
            ]
        else:
            # No fence was crossed; the alert stands on its behaviours alone.
            factors = [
                RiskFactor(
                    label=self.zoneless_label,
                    points=self.zoneless_base_risk,
                    detail="threat to the camera itself, outside any virtual fence",
                )
            ]

        confidence_points = 0
        if detection is not None:
            confidence_points = int(
                round((detection.confidence - self.confidence_pivot) * self.confidence_gain)
            )
            confidence_points = max(
                -self.confidence_cap, min(self.confidence_cap, confidence_points)
            )
        if confidence_points:
            factors.append(
                RiskFactor(
                    label="DETECTION CONFIDENCE",
                    points=confidence_points,
                    detail=f"model confidence {detection.confidence:.0%}",
                )
            )

        if context.is_night:
            factors.append(
                RiskFactor(
                    label="NIGHT-TIME",
                    points=self.night_bonus,
                    detail=f"movement at {context.timestamp:%H:%M} (night window)",
                )
            )

        for behaviour in behaviours or []:
            factors.append(
                RiskFactor(
                    label=behaviour.name, points=behaviour.points, detail=behaviour.detail
                )
            )

        score = max(0, min(100, sum(f.points for f in factors)))
        return RiskAssessment(
            score=score,
            severity=self.severity(score),
            reason=self._reason(factors, zone, context),
            factors=factors,
        )

    def severity(self, score: int) -> str:
        critical, high, medium = self.thresholds
        if score >= critical:
            return CRITICAL
        if score >= high:
            return HIGH
        if score >= medium:
            return MEDIUM
        return LOW

    @staticmethod
    def _reason(factors: list[RiskFactor], zone: Zone | None, context: SceneContext) -> str:
        aggravating = [f for f in factors[1:] if f.points > 0]
        base = (
            f"{zone.kind.title()}-zone entry detected"
            if zone is not None
            else "Camera integrity threat detected"
        )
        if not aggravating:
            return f"{base}."
        if len(aggravating) == 1:
            return f"{base}, aggravated by {aggravating[0].label.lower()}."
        return (
            f"{base}. Multiple contextual signals "
            f"({', '.join(f.label.lower() for f in aggravating)}) indicate suspicious activity."
        )
