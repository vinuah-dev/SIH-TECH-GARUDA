"""Behaviour Engine: naming what the Context Engine measures."""

import pytest

from app.behaviour.engine import (
    BORDER_FACING,
    ERRATIC,
    LOITERING,
    NIGHT_MOVEMENT,
    BehaviourEngine,
)
from app.context.engine import ContextEngine
from app.context.scene import SceneContext
from app.zones.manager import ZoneManager

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
DAY = SceneContext.build("CAM-01", force_night=False)


@pytest.fixture
def zones():
    return ZoneManager.from_dict(CONFIG)


def run(zones, positions, scene=NIGHT, step=0.5, behaviour=None):
    """Drive context + behaviour over a path; return the final behaviour names."""
    context_engine = ContextEngine(zones)
    behaviour = behaviour or BehaviourEngine()
    names: set[str] = set()
    found = []
    for i, position in enumerate(positions):
        zone = zones.locate(position)
        track = context_engine.observe(1, position, zone, scene, i * step)
        found = behaviour.classify(track)
        names = {b.name for b in found}
    return names, found


def line(start, end, count, y=0.9):
    """`count` evenly spaced points from start to end on one horizontal line."""
    return [(start + (end - start) * i / (count - 1), y) for i in range(count)]


# ----------------------------------------------------------------- loitering


def test_loitering_needs_more_than_the_grace_period(zones):
    names, _ = run(zones, [(0.8, 0.9)] * 6, step=0.5)  # 2.5 s inside
    assert LOITERING not in names


def test_standing_inside_a_zone_becomes_loitering(zones):
    names, found = run(zones, [(0.8, 0.9)] * 30, step=1.0)  # 29 s inside
    assert LOITERING in names
    loitering = next(b for b in found if b.name == LOITERING)
    assert loitering.points > 0
    assert "RESTRICTED" in loitering.detail


def test_loitering_points_are_graded_and_capped(zones):
    engine = BehaviourEngine(loiter_after=5, loiter_per_second=1.5, loiter_cap=20)
    _, short = run(zones, [(0.8, 0.9)] * 12, step=1.0, behaviour=engine)
    _, long = run(zones, [(0.8, 0.9)] * 60, step=1.0, behaviour=BehaviourEngine())
    short_points = next(b.points for b in short if b.name == LOITERING)
    long_points = next(b.points for b in long if b.name == LOITERING)
    assert short_points < long_points
    assert long_points == 20


def test_standing_outside_every_zone_is_not_loitering(zones):
    names, _ = run(zones, [(0.1, 0.9)] * 30, step=1.0)
    assert LOITERING not in names


# ------------------------------------------------------------ border-facing


def test_a_brief_approach_is_not_border_facing(zones):
    engine = BehaviourEngine(approach_frames=5)
    names, _ = run(zones, line(0.1, 0.16, 3), behaviour=engine)
    assert BORDER_FACING not in names


def test_sustained_approach_is_border_facing(zones):
    names, found = run(zones, line(0.05, 0.5, 12))
    assert BORDER_FACING in names
    assert "RESTRICTED" in next(b for b in found if b.name == BORDER_FACING).detail


def test_walking_away_is_never_border_facing(zones):
    names, _ = run(zones, line(0.5, 0.05, 12))
    assert BORDER_FACING not in names


def test_the_approach_streak_resets_when_the_track_turns_around(zones):
    engine = BehaviourEngine(approach_frames=5)
    context_engine = ContextEngine(ZoneManager.from_dict(CONFIG))
    zones_ = context_engine.zones

    names = set()
    path = line(0.05, 0.45, 12) + line(0.45, 0.05, 6)
    for i, position in enumerate(path):
        track = context_engine.observe(1, position, zones_.locate(position), NIGHT, i * 0.5)
        names = {b.name for b in engine.classify(track)}
    assert BORDER_FACING not in names


# --------------------------------------------------------------- night-time


def test_movement_at_night_is_flagged(zones):
    names, _ = run(zones, line(0.05, 0.4, 10), scene=NIGHT)
    assert NIGHT_MOVEMENT in names


def test_movement_in_daylight_is_not_flagged(zones):
    names, _ = run(zones, line(0.05, 0.4, 10), scene=DAY)
    assert NIGHT_MOVEMENT not in names


def test_standing_still_at_night_is_not_night_movement(zones):
    names, _ = run(zones, [(0.2, 0.9)] * 10, scene=NIGHT)
    assert NIGHT_MOVEMENT not in names


# ------------------------------------------------------------------ erratic


def test_a_straight_walk_is_not_erratic(zones):
    names, _ = run(zones, line(0.05, 0.5, 15))
    assert ERRATIC not in names


def test_pacing_back_and_forth_is_erratic(zones):
    """Someone doubling back repeatedly, which a straight-line walker never does."""
    path = []
    for i in range(16):
        path.append((0.20 + (0.06 if i % 2 else 0.0), 0.9 + (0.05 if i % 4 < 2 else -0.05)))
    names, found = run(zones, path)
    assert ERRATIC in names
    assert next(b for b in found if b.name == ERRATIC).points > 0


def test_erratic_needs_enough_samples(zones):
    engine = BehaviourEngine(erratic_min_samples=8)
    names, _ = run(zones, [(0.2, 0.9), (0.26, 0.95), (0.2, 0.9)], behaviour=engine)
    assert ERRATIC not in names


# -------------------------------------------------------------- bookkeeping


def test_behaviours_are_tracked_independently(zones):
    engine = BehaviourEngine(approach_frames=3)
    context_engine = ContextEngine(zones)
    for i, position in enumerate(line(0.05, 0.5, 10)):
        context_engine.observe(1, position, zones.locate(position), NIGHT, i * 0.5)
        engine.classify(context_engine.observe(1, position, zones.locate(position), NIGHT, i * 0.5))
    # Track 2 has only just appeared and must not inherit track 1's streak.
    track2 = context_engine.observe(2, (0.05, 0.9), None, NIGHT, 5.0)
    assert BORDER_FACING not in {b.name for b in engine.classify(track2)}


def test_forget_clears_track_state(zones):
    engine = BehaviourEngine(approach_frames=3)
    context_engine = ContextEngine(zones)
    for i, position in enumerate(line(0.05, 0.5, 10)):
        engine.classify(context_engine.observe(1, position, zones.locate(position), NIGHT, i * 0.5))
    engine.forget(1)
    assert engine._approach_streak == {}
