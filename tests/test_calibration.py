"""Ground-plane calibration: image pixels -> real-world metres."""

import math

import pytest

from app.behaviour.engine import RUNNING, BehaviourEngine
from app.calibration import GroundPlane
from app.context.engine import ContextEngine
from app.context.scene import SceneContext
from app.zones.manager import ZoneManager

# A 10 m x 10 m ground square filling the frame, so the maths is checkable by hand.
SQUARE = [
    ((0.0, 0.0), (0.0, 10.0)),
    ((1.0, 0.0), (10.0, 10.0)),
    ((1.0, 1.0), (10.0, 0.0)),
    ((0.0, 1.0), (0.0, 0.0)),
]

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
CALIBRATION = {
    "unit": "m",
    "points": [
        {"image": list(img), "world": list(world)} for img, world in SQUARE
    ],
}
NIGHT = SceneContext.build("CAM-01", force_night=True)


@pytest.fixture
def plane():
    return GroundPlane.from_points(SQUARE)


# ------------------------------------------------------------- construction


def test_requires_exactly_four_points():
    with pytest.raises(ValueError, match="exactly 4"):
        GroundPlane.from_points(SQUARE[:3])


def test_rejects_duplicate_image_points():
    bad = [SQUARE[0], SQUARE[0], SQUARE[2], SQUARE[3]]
    with pytest.raises(ValueError, match="distinct"):
        GroundPlane.from_points(bad)


def test_builds_from_config_dict():
    plane = GroundPlane.from_config(CALIBRATION)
    assert plane.unit == "m"
    assert plane.project((0.0, 1.0)) == pytest.approx((0.0, 0.0), abs=1e-4)


# ----------------------------------------------------------------- geometry


def test_corners_project_to_their_surveyed_positions(plane):
    for image, world in SQUARE:
        assert plane.project(image) == pytest.approx(world, abs=1e-4)


def test_frame_centre_projects_to_the_middle_of_the_square(plane):
    assert plane.project((0.5, 0.5)) == pytest.approx((5.0, 5.0), abs=1e-4)


def test_distance_is_measured_in_world_units(plane):
    assert plane.distance((0.0, 1.0), (1.0, 1.0)) == pytest.approx(10.0, abs=1e-4)
    assert plane.distance((0.0, 1.0), (1.0, 0.0)) == pytest.approx(math.hypot(10, 10), abs=1e-3)


def test_equal_pixel_steps_can_cover_unequal_ground():
    """The whole point of a homography: perspective is not linear."""
    trapezoid = [
        ((0.2, 0.4), (0.0, 50.0)),
        ((0.8, 0.4), (20.0, 50.0)),
        ((1.0, 1.0), (20.0, 5.0)),
        ((0.0, 1.0), (0.0, 5.0)),
    ]
    plane = GroundPlane.from_points(trapezoid)
    near = plane.distance((0.4, 1.0), (0.5, 1.0))
    far = plane.distance((0.4, 0.4), (0.5, 0.4))
    assert far > near * 1.5


# ------------------------------------------------------------ configuration


def test_zone_config_without_calibration_has_no_ground_plane():
    assert ZoneManager.from_dict(CONFIG).ground is None


def test_zone_config_with_calibration_loads_a_ground_plane():
    manager = ZoneManager.from_dict({**CONFIG, "calibration": CALIBRATION})
    assert manager.ground is not None


def test_shipped_calibrated_config_loads():
    manager = ZoneManager.from_file("config/zones-calibrated.json")
    assert manager.ground is not None
    assert {z.name for z in manager} == {"RESTRICTED", "WATCH"}


# --------------------------------------------------------- context reporting


def observe(manager, positions, step=1.0):
    engine = ContextEngine(manager)
    track = None
    for i, position in enumerate(positions):
        track = engine.observe(1, position, manager.locate(position), NIGHT, i * step)
    return track


def test_uncalibrated_context_reports_no_metric_values():
    track = observe(ZoneManager.from_dict(CONFIG), [(0.1, 0.9), (0.2, 0.9)])
    assert track.speed_mps is None
    assert track.distance_to_target_m is None
    assert not track.calibrated
    assert track.speed_label.endswith("w/s")


def test_calibrated_context_reports_metres_per_second():
    manager = ZoneManager.from_dict({**CONFIG, "calibration": CALIBRATION})
    # 0.1 of frame width per second, across a 10 m square = 1 m/s.
    track = observe(manager, [(0.1, 0.9), (0.2, 0.9)])
    assert track.calibrated
    assert track.speed_mps == pytest.approx(1.0, abs=1e-3)
    assert track.speed_label == "1.00 m/s"


def test_calibrated_context_reports_distance_in_metres():
    manager = ZoneManager.from_dict({**CONFIG, "calibration": CALIBRATION})
    # Zone edge is at x=0.6; standing at x=0.1 is 0.5 widths = 5 m away.
    track = observe(manager, [(0.1, 0.5), (0.1, 0.5)])
    assert track.distance_to_target_m == pytest.approx(5.0, abs=1e-3)


def test_units_do_not_flip_on_the_first_frame():
    manager = ZoneManager.from_dict({**CONFIG, "calibration": CALIBRATION})
    track = observe(manager, [(0.1, 0.9)])
    assert track.calibrated
    assert track.speed_label == "0.00 m/s"


def test_movement_threshold_uses_metres_when_calibrated():
    manager = ZoneManager.from_dict({**CONFIG, "calibration": CALIBRATION})
    # 0.02 widths/s = 0.2 m/s: above the frame-unit threshold, below the metric one.
    track = observe(manager, [(0.10, 0.9), (0.12, 0.9)])
    assert track.speed > 0.01
    assert track.speed_mps < 0.25
    assert not track.is_moving


# ------------------------------------------------------- calibrated behaviour


def test_rapid_movement_needs_calibration():
    """Speed alone cannot be judged without knowing what a pixel is worth."""
    track = observe(ZoneManager.from_dict(CONFIG), [(0.1, 0.9), (0.9, 0.9)])
    assert RUNNING not in {b.name for b in BehaviourEngine().classify(track)}


def test_rapid_movement_fires_above_the_speed_threshold():
    manager = ZoneManager.from_dict({**CONFIG, "calibration": CALIBRATION})
    # 0.3 widths/s across a 10 m square = 3 m/s.
    track = observe(manager, [(0.1, 0.9), (0.4, 0.9)])
    behaviours = {b.name for b in BehaviourEngine(running_speed_mps=2.0).classify(track)}
    assert RUNNING in behaviours


def test_a_walking_pace_is_not_rapid_movement():
    manager = ZoneManager.from_dict({**CONFIG, "calibration": CALIBRATION})
    track = observe(manager, [(0.1, 0.9), (0.2, 0.9)])  # 1 m/s
    assert RUNNING not in {b.name for b in BehaviourEngine().classify(track)}
