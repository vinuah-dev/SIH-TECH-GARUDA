"""Camera tampering: someone close enough to interfere with the camera itself.

Cameras are mounted high and look outward, so a person filling the frame is
physically at the lens. This is a zone-independent threat - blinding the camera
defeats every zone rule - so it has its own trigger and its own alert.
"""

import pytest

from app.alerts.events import IntrusionMonitor
from app.behaviour.engine import APPROACHING_CAMERA, CAMERA_TAMPERING, BehaviourEngine
from app.context.engine import ContextEngine
from app.context.scene import SceneContext
from app.zones.manager import Zone, ZoneManager

CONFIG = {
    "zones": [
        {
            "name": "RESTRICTED",
            "kind": "RESTRICTED",
            "base_risk": 60,
            "polygon": [[0.6, 0.0], [1.0, 0.0], [1.0, 1.0], [0.6, 1.0]],
        }
    ]
}
NIGHT = SceneContext.build("CAM-01", force_night=True)
RESTRICTED = Zone("RESTRICTED", "RESTRICTED", [(0.6, 0.0), (1.0, 0.0), (1.0, 1.0)], base_risk=60)


def box(height_ratio, bottom=0.95):
    """A person bbox of the given height fraction, sitting on `bottom`."""
    return (0.45, max(0.0, bottom - height_ratio), 0.55, bottom)


def walk(heights, position=(0.2, 0.9), step=0.5, bottoms=None):
    """Feed a sequence of apparent sizes; return the final behaviour names."""
    zones = ZoneManager.from_dict(CONFIG)
    context_engine = ContextEngine(zones)
    behaviour = BehaviourEngine()
    found = []
    for i, ratio in enumerate(heights):
        bottom = bottoms[i] if bottoms else 0.95
        track = context_engine.observe(
            1, position, zones.locate(position), NIGHT, i * step,
            bbox_norm=box(ratio, bottom),
        )
        found = behaviour.classify(track)
    return {b.name for b in found}, found


# ------------------------------------------------------------- measurement


def test_apparent_size_is_measured_from_the_bbox():
    zones = ZoneManager.from_dict(CONFIG)
    engine = ContextEngine(zones)
    track = engine.observe(1, (0.2, 0.9), None, NIGHT, 0.0, bbox_norm=(0.4, 0.2, 0.6, 0.9))
    assert track.height_ratio == pytest.approx(0.7)
    assert track.coverage == pytest.approx(0.2 * 0.7)


def test_a_box_reaching_the_bottom_edge_is_flagged_as_clipped():
    """Feet out of shot is what standing under the camera looks like."""
    zones = ZoneManager.from_dict(CONFIG)
    engine = ContextEngine(zones)
    clipped = engine.observe(1, (0.2, 0.9), None, NIGHT, 0.0, bbox_norm=(0.4, 0.1, 0.6, 1.0))
    assert clipped.bottom_clipped
    normal = engine.observe(2, (0.2, 0.9), None, NIGHT, 0.0, bbox_norm=(0.4, 0.1, 0.6, 0.8))
    assert not normal.bottom_clipped


def test_growth_rate_is_positive_when_closing_on_the_camera():
    zones = ZoneManager.from_dict(CONFIG)
    engine = ContextEngine(zones)
    for i, ratio in enumerate([0.2, 0.3, 0.4]):
        track = engine.observe(1, (0.2, 0.9), None, NIGHT, i * 1.0, bbox_norm=box(ratio))
    assert track.camera_growth == pytest.approx(0.1, abs=1e-6)


def test_growth_rate_is_negative_when_walking_away():
    zones = ZoneManager.from_dict(CONFIG)
    engine = ContextEngine(zones)
    for i, ratio in enumerate([0.6, 0.4, 0.2]):
        track = engine.observe(1, (0.2, 0.9), None, NIGHT, i * 1.0, bbox_norm=box(ratio))
    assert track.camera_growth < 0


def test_no_bbox_means_no_size_signal():
    """Detectors that do not supply a box must not trigger tampering."""
    zones = ZoneManager.from_dict(CONFIG)
    track = ContextEngine(zones).observe(1, (0.2, 0.9), None, NIGHT, 0.0)
    assert track.height_ratio == 0.0
    assert not track.near_camera


# --------------------------------------------------------------- behaviour


def test_a_distant_person_is_not_tampering():
    names, _ = walk([0.2] * 6)
    assert CAMERA_TAMPERING not in names
    assert APPROACHING_CAMERA not in names


def test_a_person_filling_the_frame_is_tampering():
    names, found = walk([0.70] * 6)
    assert CAMERA_TAMPERING in names
    assert next(b for b in found if b.name == CAMERA_TAMPERING).points > 0


def test_steady_growth_toward_the_camera_is_flagged_before_tampering():
    names, _ = walk([0.30, 0.36, 0.42, 0.48])
    assert APPROACHING_CAMERA in names
    assert CAMERA_TAMPERING not in names


def test_tampering_supersedes_the_approach_signal():
    """Once they are at the lens, a growth signal is redundant noise."""
    names, _ = walk([0.30, 0.45, 0.60, 0.75])
    assert CAMERA_TAMPERING in names
    assert APPROACHING_CAMERA not in names


def test_a_large_but_stable_person_is_not_approaching():
    """A camera on a low mount sees people large all the time."""
    names, _ = walk([0.45] * 8)
    assert APPROACHING_CAMERA not in names


def test_clipped_feet_are_mentioned_in_the_alert_detail():
    _, found = walk([0.75] * 4, bottoms=[1.0] * 4)
    detail = next(b for b in found if b.name == CAMERA_TAMPERING).detail
    assert "feet out of shot" in detail


def test_tampering_outweighs_every_other_behaviour():
    engine = BehaviourEngine()
    assert engine.tamper_points > engine.loiter_cap
    assert engine.tamper_points > engine.border_facing_points


def test_threshold_is_configurable():
    zones = ZoneManager.from_dict(CONFIG)
    context = ContextEngine(zones).observe(
        1, (0.2, 0.9), None, NIGHT, 0.0, bbox_norm=box(0.50)
    )
    assert CAMERA_TAMPERING not in {b.name for b in BehaviourEngine().classify(context)}
    lenient = BehaviourEngine(tamper_height_ratio=0.40)
    assert CAMERA_TAMPERING in {b.name for b in lenient.classify(context)}


# ---------------------------------------------------------------- alerting


def test_tampering_alerts_without_any_zone():
    """The whole point: no fence has been crossed, but this is still an alert."""
    monitor = IntrusionMonitor(confirm_frames=2, cooldown_seconds=30)
    assert not monitor.update_tampering(1, 0.0, True)
    assert monitor.update_tampering(1, 1.0, True)


def test_tampering_needs_confirmation_frames_too():
    monitor = IntrusionMonitor(confirm_frames=3, cooldown_seconds=30)
    assert [monitor.update_tampering(1, float(t), True) for t in range(4)] == [
        False, False, True, False
    ]


def test_the_tamper_counter_resets_when_they_back_off():
    monitor = IntrusionMonitor(confirm_frames=3, cooldown_seconds=30)
    monitor.update_tampering(1, 0.0, True)
    monitor.update_tampering(1, 1.0, True)
    monitor.update_tampering(1, 2.0, False)
    assert not monitor.update_tampering(1, 3.0, True)


def test_zone_and_tamper_alerts_are_independent():
    """Being inside a fence must not consume the tampering confirmation."""
    monitor = IntrusionMonitor(confirm_frames=2, cooldown_seconds=30)
    monitor.update(1, RESTRICTED, 0.0)
    monitor.update_tampering(1, 0.0, True)
    assert monitor.update(1, RESTRICTED, 1.0)
    assert monitor.update_tampering(1, 1.0, True)


def test_tampering_respects_the_cooldown():
    monitor = IntrusionMonitor(confirm_frames=1, cooldown_seconds=30)
    assert monitor.update_tampering(1, 0.0, True)
    assert not monitor.update_tampering(1, 5.0, True)
    assert monitor.update_tampering(1, 31.0, True)


def test_untracked_detections_never_raise_tampering():
    assert not IntrusionMonitor(confirm_frames=1).update_tampering(None, 0.0, True)


# -------------------------------------------------------------------- risk


def test_a_zoneless_alert_is_scored_on_its_behaviours():
    from app.behaviour.engine import Behaviour
    from app.detection.base import Detection
    from app.risk.engine import RiskEngine

    detection = Detection("PERSON", 0.94, (10, 10, 60, 400), track_id=1)
    assessment = RiskEngine().assess(
        detection, None, NIGHT, None,
        [Behaviour(CAMERA_TAMPERING, 30, "person fills 75% of frame height")],
    )
    assert assessment.severity == "CRITICAL"
    assert "Camera integrity threat" in assessment.reason
    assert any(f.label == "CAMERA INTEGRITY" for f in assessment.factors)
    assert sum(f.points for f in assessment.factors) == assessment.score
