"""The intrusion state machine decides when occupancy becomes an alert."""

from app.alerts.events import IntrusionMonitor
from app.zones.manager import Zone

RESTRICTED = Zone("RESTRICTED", "RESTRICTED", [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0)], base_risk=60)
WATCH = Zone("WATCH", "WATCH", [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0)], base_risk=30)


def test_alert_requires_consecutive_confirmation_frames():
    monitor = IntrusionMonitor(confirm_frames=3, cooldown_seconds=30)
    assert [monitor.update(1, RESTRICTED, t) for t in range(4)] == [False, False, True, False]


def test_single_frame_flicker_does_not_alert():
    """One stray in-zone frame must not raise an alert."""
    monitor = IntrusionMonitor(confirm_frames=3, cooldown_seconds=30)
    monitor.update(1, RESTRICTED, 0)
    monitor.update(1, None, 1)
    monitor.update(1, RESTRICTED, 2)
    monitor.update(1, None, 3)
    assert not monitor.update(1, RESTRICTED, 4)


def test_observation_only_zones_never_alert():
    monitor = IntrusionMonitor(confirm_frames=1, alert_kinds=frozenset({"RESTRICTED"}))
    assert not any(monitor.update(1, WATCH, t) for t in range(10))


def test_cooldown_suppresses_repeat_alerts_for_the_same_track():
    monitor = IntrusionMonitor(confirm_frames=1, cooldown_seconds=30)
    assert monitor.update(1, RESTRICTED, 0)
    assert not any(monitor.update(1, RESTRICTED, t) for t in range(1, 30))


def test_alert_repeats_once_the_cooldown_expires():
    monitor = IntrusionMonitor(confirm_frames=1, cooldown_seconds=30)
    monitor.update(1, RESTRICTED, 0)
    assert monitor.update(1, RESTRICTED, 31)


def test_each_track_is_scored_independently():
    monitor = IntrusionMonitor(confirm_frames=1, cooldown_seconds=30)
    assert monitor.update(1, RESTRICTED, 0)
    assert monitor.update(2, RESTRICTED, 0)


def test_moving_between_zones_restarts_confirmation():
    monitor = IntrusionMonitor(confirm_frames=2, cooldown_seconds=30)
    monitor.update(1, WATCH, 0)
    monitor.update(1, WATCH, 1)
    assert not monitor.update(1, RESTRICTED, 2)
    assert monitor.update(1, RESTRICTED, 3)


def test_untracked_detections_never_alert():
    monitor = IntrusionMonitor(confirm_frames=1)
    assert not monitor.update(None, RESTRICTED, 0)


def test_stale_tracks_are_forgotten():
    monitor = IntrusionMonitor(confirm_frames=1, cooldown_seconds=30, forget_after=60)
    monitor.update(1, RESTRICTED, 0)
    # Track vanishes for longer than forget_after, then a new person reuses ID 1.
    assert monitor.update(1, RESTRICTED, 200)


def test_escalation_bypasses_the_cooldown():
    """A worsening situation must reach the operator even during a cooldown."""
    monitor = IntrusionMonitor(confirm_frames=1, cooldown_seconds=30)
    assert monitor.update(1, RESTRICTED, 0, "HIGH")
    assert not monitor.update(1, RESTRICTED, 1, "HIGH")
    assert monitor.update(1, RESTRICTED, 2, "CRITICAL")


def test_de_escalation_does_not_re_alert():
    monitor = IntrusionMonitor(confirm_frames=1, cooldown_seconds=30)
    monitor.update(1, RESTRICTED, 0, "CRITICAL")
    assert not monitor.update(1, RESTRICTED, 1, "HIGH")


def test_repeated_escalation_alerts_only_once_per_step():
    monitor = IntrusionMonitor(confirm_frames=1, cooldown_seconds=30)
    monitor.update(1, RESTRICTED, 0, "MEDIUM")
    assert monitor.update(1, RESTRICTED, 1, "HIGH")
    assert not monitor.update(1, RESTRICTED, 2, "HIGH")
    assert monitor.update(1, RESTRICTED, 3, "CRITICAL")


def test_severity_is_optional_and_preserves_plain_cooldown():
    monitor = IntrusionMonitor(confirm_frames=1, cooldown_seconds=30)
    assert monitor.update(1, RESTRICTED, 0)
    assert not monitor.update(1, RESTRICTED, 1)
