"""Context Engine: speed, heading, dwell and approach detection."""

import pytest

from app.context.engine import ContextEngine, TrackContext
from app.context.history import TrackHistory
from app.context.scene import SceneContext
from app.zones.geometry import distance_to_polygon, point_to_segment_distance
from app.zones.manager import ZoneManager

# RESTRICTED occupies the right half of the frame.
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

SCENE = SceneContext.build("CAM-01", force_night=True)


@pytest.fixture
def engine():
    return ContextEngine(ZoneManager.from_dict(CONFIG))


def walk(engine, positions, start=0.0, step=0.5):
    """Feed a sequence of positions one step apart; return the last context."""
    context = None
    for i, position in enumerate(positions):
        now = start + i * step
        zone = engine.zones.locate(position)
        context = engine.observe(1, position, zone, SCENE, now)
    return context


# ------------------------------------------------------------------ geometry


def test_point_to_segment_distance_clamps_to_the_ends():
    assert point_to_segment_distance((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)) == 1.0
    assert point_to_segment_distance((1.5, 1.0), (1.0, 0.0), (2.0, 0.0)) == 1.0


def test_distance_to_polygon_is_zero_inside():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert distance_to_polygon((0.5, 0.5), square) == 0.0
    assert distance_to_polygon((1.5, 0.5), square) == pytest.approx(0.5)


# --------------------------------------------------------------------- speed


def test_first_observation_has_no_speed_or_heading(engine):
    context = walk(engine, [(0.1, 0.9)])
    assert context.speed == 0.0
    assert context.heading == "-"
    assert not context.is_moving


def test_speed_is_normalized_units_per_second(engine):
    # 0.1 width per 0.5 s step = 0.2 per second.
    context = walk(engine, [(0.1, 0.9), (0.2, 0.9), (0.3, 0.9)])
    assert context.speed == pytest.approx(0.2, abs=1e-6)


def test_speed_is_independent_of_frame_rate():
    zones = ZoneManager.from_dict(CONFIG)
    slow = walk(ContextEngine(zones), [(0.1, 0.9), (0.3, 0.9)], step=1.0)
    fast = walk(ContextEngine(zones), [(0.1, 0.9), (0.2, 0.9), (0.3, 0.9)], step=0.5)
    assert slow.speed == pytest.approx(fast.speed, abs=1e-6)


def test_a_stationary_track_reports_no_heading(engine):
    context = walk(engine, [(0.2, 0.9)] * 5)
    assert context.heading == "-"
    assert not context.is_moving


# ------------------------------------------------------------------- heading


@pytest.mark.parametrize(
    "start,end,heading",
    [
        ((0.1, 0.5), (0.4, 0.5), "E"),
        ((0.4, 0.5), (0.1, 0.5), "W"),
        ((0.5, 0.8), (0.5, 0.4), "N"),  # image y grows downward
        ((0.5, 0.4), (0.5, 0.8), "S"),
    ],
)
def test_heading_uses_screen_coordinates(engine, start, end, heading):
    assert walk(engine, [start, end]).heading == heading


# --------------------------------------------------------------------- dwell


def test_dwell_accumulates_while_inside_one_zone(engine):
    context = walk(engine, [(0.7, 0.9)] * 5, step=1.0)  # 4 s of footage
    assert context.zone.name == "RESTRICTED"
    assert context.dwell_seconds == pytest.approx(4.0)


def test_dwell_resets_when_the_zone_changes(engine):
    engine.observe(1, (0.7, 0.9), engine.zones.locate((0.7, 0.9)), SCENE, 0.0)
    engine.observe(1, (0.7, 0.9), engine.zones.locate((0.7, 0.9)), SCENE, 5.0)
    context = engine.observe(1, (0.1, 0.9), None, SCENE, 6.0)
    assert context.dwell_seconds == 0.0


def test_dwell_is_zero_outside_every_zone(engine):
    assert walk(engine, [(0.1, 0.9)] * 4).dwell_seconds == 0.0


# ----------------------------------------------------------------- approach


def test_closing_on_the_zone_is_flagged(engine):
    context = walk(engine, [(0.1, 0.9), (0.25, 0.9), (0.4, 0.9)])
    assert context.approaching
    assert context.target_zone == "RESTRICTED"
    assert context.distance_to_target == pytest.approx(0.2)


def test_walking_away_is_not_approaching(engine):
    assert not walk(engine, [(0.4, 0.9), (0.25, 0.9), (0.1, 0.9)]).approaching


def test_jitter_below_the_epsilon_is_not_approaching(engine):
    context = walk(engine, [(0.100, 0.9), (0.103, 0.9), (0.101, 0.9)])
    assert not context.approaching


def test_no_armed_zone_means_no_target():
    zones = ZoneManager.from_dict(CONFIG)
    engine = ContextEngine(zones, armed_kinds=frozenset({"PATROL"}))
    context = walk(engine, [(0.1, 0.9), (0.4, 0.9)])
    assert context.target_zone is None
    assert not context.approaching


# ------------------------------------------------------------------ history


def test_tracks_are_independent(engine):
    engine.observe(1, (0.1, 0.9), None, SCENE, 0.0)
    engine.observe(2, (0.8, 0.9), engine.zones.locate((0.8, 0.9)), SCENE, 0.0)
    assert engine.history.get(1).current_zone is None
    assert engine.history.get(2).current_zone == "RESTRICTED"


def test_history_window_is_bounded():
    history = TrackHistory(window=5)
    for i in range(20):
        history.observe(1, (i / 100, 0.9), None, float(i))
    assert len(history.get(1).points) == 5


def test_stale_tracks_are_forgotten():
    history = TrackHistory(forget_after=10)
    history.observe(1, (0.1, 0.9), None, 0.0)
    history.observe(2, (0.2, 0.9), None, 100.0)
    assert history.get(1) is None
    assert history.get(2) is not None


def test_context_serializes_for_the_event_log(engine):
    context = walk(engine, [(0.1, 0.9), (0.4, 0.9)])
    payload = context.to_dict()
    assert set(payload) == {
        "speed", "heading", "dwell_seconds", "track_age_seconds",
        "distance_to_target", "target_zone", "approaching", "heading_variance", "samples",
        "calibrated", "speed_mps", "distance_to_target_m",
        "height_ratio", "coverage", "camera_growth", "bottom_clipped",
    }
    assert payload["calibrated"] is False
    assert payload["speed_mps"] is None
    assert isinstance(context, TrackContext)
