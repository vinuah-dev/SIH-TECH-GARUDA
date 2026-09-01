"""Camera integrity: a feed that stops answering may have been tampered with.

Cutting a cable, unplugging the recorder and knocking the housing off its mount
all look the same from software - the feed stops - and all three are things
done *before* crossing a fence so that nothing records it. These tests pin down
which link transitions are noise and which are security events.
"""

import pytest

from app.behaviour.camera import (
    CAMERA_OFFLINE,
    LINK_LOST,
    REPEATED_LINK_LOSS,
    CameraIntegrityMonitor,
)
from app.context.scene import SceneContext
from app.risk.engine import RiskEngine
from app.video.stream import CameraState

NIGHT = SceneContext.build("CAM-01", force_night=True)
DAY = SceneContext.build("CAM-01", force_night=False)


def names(behaviours):
    return {b.name for b in behaviours}


@pytest.fixture
def monitor():
    return CameraIntegrityMonitor()


# ------------------------------------------------------------- brief blips


def test_a_short_outage_that_heals_is_not_an_event(monitor):
    """Networks glitch. Alerting on every blip trains operators to ignore alerts."""
    assert monitor.observe(CameraState.RECONNECTING, 0) == []
    assert monitor.observe(CameraState.ONLINE, 5) == []


def test_a_healthy_camera_never_raises_anything(monitor):
    assert monitor.observe(CameraState.CONNECTING, 0) == []
    assert monitor.observe(CameraState.ONLINE, 1) == []
    assert monitor.observe(CameraState.ONLINE, 2) == []


def test_recovery_without_a_recorded_drop_is_ignored(monitor):
    assert monitor.observe(CameraState.ONLINE, 10) == []


# ----------------------------------------------------------- real outages


def test_a_long_outage_that_recovers_is_reported(monitor):
    monitor.observe(CameraState.RECONNECTING, 0)
    found = monitor.observe(CameraState.ONLINE, 45)
    assert LINK_LOST in names(found)
    assert "45s" in found[0].detail


def test_the_grace_period_is_the_boundary():
    monitor = CameraIntegrityMonitor(outage_grace=10.0)
    monitor.observe(CameraState.RECONNECTING, 0)
    assert monitor.observe(CameraState.ONLINE, 9.9) == []

    monitor = CameraIntegrityMonitor(outage_grace=10.0)
    monitor.observe(CameraState.RECONNECTING, 0)
    assert LINK_LOST in names(monitor.observe(CameraState.ONLINE, 10.0))


def test_a_camera_that_never_returns_is_the_most_serious_finding(monitor):
    monitor.observe(CameraState.RECONNECTING, 0)
    found = monitor.observe(CameraState.OFFLINE, 90)
    assert CAMERA_OFFLINE in names(found)
    offline = next(b for b in found if b.name == CAMERA_OFFLINE)
    assert offline.points > monitor.link_lost_points
    assert "not recovering" in offline.detail


def test_going_offline_without_a_prior_drop_is_still_reported(monitor):
    found = monitor.observe(CameraState.OFFLINE, 5)
    assert CAMERA_OFFLINE in names(found)


# -------------------------------------------------------- interference


def test_repeated_drops_are_reported_as_a_pattern():
    """Three short drops are not three glitches; somebody is interfering."""
    monitor = CameraIntegrityMonitor(repeat_threshold=3)
    seen = []
    at = 0.0
    for _ in range(3):
        seen += monitor.observe(CameraState.RECONNECTING, at)
        monitor.observe(CameraState.ONLINE, at + 2)
        at += 30
    assert REPEATED_LINK_LOSS in names(seen)


def test_two_drops_are_below_the_threshold():
    monitor = CameraIntegrityMonitor(repeat_threshold=3)
    seen = []
    for at in (0.0, 30.0):
        seen += monitor.observe(CameraState.RECONNECTING, at)
        monitor.observe(CameraState.ONLINE, at + 2)
    assert REPEATED_LINK_LOSS not in names(seen)


def test_drops_spread_beyond_the_window_are_not_a_pattern():
    monitor = CameraIntegrityMonitor(repeat_threshold=3, repeat_window=60.0)
    seen = []
    at = 0.0
    for _ in range(4):
        seen += monitor.observe(CameraState.RECONNECTING, at)
        monitor.observe(CameraState.ONLINE, at + 1)
        at += 100  # each drop ages out before the next
    assert REPEATED_LINK_LOSS not in names(seen)


# ---------------------------------------------------------------- cooldown


def test_the_same_finding_is_not_repeated_while_it_stays_true():
    monitor = CameraIntegrityMonitor(cooldown=60.0)
    assert CAMERA_OFFLINE in names(monitor.observe(CameraState.OFFLINE, 0))
    assert monitor.observe(CameraState.OFFLINE, 10) == []


def test_the_finding_can_be_raised_again_after_the_cooldown():
    monitor = CameraIntegrityMonitor(cooldown=60.0)
    monitor.observe(CameraState.OFFLINE, 0)
    assert CAMERA_OFFLINE in names(monitor.observe(CameraState.OFFLINE, 61))


def test_reset_clears_all_state(monitor):
    monitor.observe(CameraState.RECONNECTING, 0)
    monitor.observe(CameraState.OFFLINE, 60)
    monitor.reset()
    assert CAMERA_OFFLINE in names(monitor.observe(CameraState.OFFLINE, 61))


# -------------------------------------------------------------------- risk


def test_an_unrecoverable_camera_scores_critical():
    monitor = CameraIntegrityMonitor()
    monitor.observe(CameraState.RECONNECTING, 0)
    behaviours = monitor.observe(CameraState.OFFLINE, 90)

    assessment = RiskEngine().assess(None, None, NIGHT, None, behaviours)
    assert assessment.severity == "CRITICAL"
    # A blinded camera saturates the scale; the factors still explain it fully.
    assert assessment.score == min(100, sum(f.points for f in assessment.factors))
    assert assessment.score == 100


def test_scoring_works_with_no_person_and_no_zone():
    """No detection means no confidence factor, and nothing should crash."""
    monitor = CameraIntegrityMonitor()
    monitor.observe(CameraState.RECONNECTING, 0)
    behaviours = monitor.observe(CameraState.ONLINE, 45)

    assessment = RiskEngine().assess(None, None, DAY, None, behaviours)
    labels = [f.label for f in assessment.factors]
    assert "DETECTION CONFIDENCE" not in labels
    assert "CAMERA INTEGRITY" in labels
    assert assessment.score > 0


def test_a_night_outage_outranks_the_same_outage_by_day():
    monitor_night = CameraIntegrityMonitor()
    monitor_night.observe(CameraState.RECONNECTING, 0)
    night = RiskEngine().assess(
        None, None, NIGHT, None, monitor_night.observe(CameraState.ONLINE, 45)
    )

    monitor_day = CameraIntegrityMonitor()
    monitor_day.observe(CameraState.RECONNECTING, 0)
    day = RiskEngine().assess(
        None, None, DAY, None, monitor_day.observe(CameraState.ONLINE, 45)
    )
    assert night.score > day.score


# ------------------------------------------------------------------- event


def test_a_camera_event_serializes_without_a_person_or_zone():
    from datetime import datetime

    from app.alerts.events import IntrusionEvent

    monitor = CameraIntegrityMonitor()
    behaviours = monitor.observe(CameraState.OFFLINE, 0)
    event = IntrusionEvent(
        event_id="CAM1",
        timestamp=datetime(2026, 8, 25, 23, 0),
        camera_id="CAM-05",
        event_type=CAMERA_OFFLINE,
        detection=None,
        zone=None,
        risk=RiskEngine().assess(None, None, NIGHT, None, behaviours),
        frame_index=0,
        behaviours=behaviours,
    )
    payload = event.to_dict()
    assert payload["object"] is None
    assert payload["zone"] is None
    assert payload["event_type"] == CAMERA_OFFLINE
    assert payload["behaviours"][0]["name"] == CAMERA_OFFLINE


def test_a_camera_event_is_storable(tmp_path):
    from datetime import datetime

    from app.alerts.events import IntrusionEvent
    from app.store import EventStore

    monitor = CameraIntegrityMonitor()
    behaviours = monitor.observe(CameraState.OFFLINE, 0)
    event = IntrusionEvent(
        event_id="CAM2",
        timestamp=datetime(2026, 8, 25, 23, 0),
        camera_id="CAM-05",
        event_type=CAMERA_OFFLINE,
        detection=None,
        zone=None,
        risk=RiskEngine().assess(None, None, NIGHT, None, behaviours),
        frame_index=0,
        behaviours=behaviours,
    )
    with EventStore(tmp_path / "cam.db") as store:
        store.write(event)
        stored = store.get("CAM2")
        assert stored["object"] is None
        assert store.summary()["by_camera"] == {"CAM-05": 1}
