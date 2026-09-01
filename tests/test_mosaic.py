"""One window for the fleet, not one window per camera.

Four cameras used to mean four windows to find, drag apart and keep on top,
and that attention is spent at exactly the moment an operator should be
watching a fence. A control room shows a wall.

Most of these tests are about the wall staying *still*: cameras arrive at
different rates, and a layout that reshuffled as frames landed would move a
camera under the operator's cursor mid-glance.
"""

import numpy as np
import pytest

from app.video.mosaic import compose, grid_shape


def frame(width=320, height=240, value=90):
    image = np.full((height, width, 3), value, np.uint8)
    image[height // 4:height // 2, width // 4:width // 2] = 255 - value
    return image


# ------------------------------------------------------------------- layout


def test_one_camera_is_shown_as_itself():
    """A single camera needs no wall, and must not be shrunk into a tile."""
    only = frame()
    assert compose({"CAM-01": only}, ["CAM-01"]) is only


@pytest.mark.parametrize("count,expected", [
    (1, (1, 1)),
    (2, (1, 2)),          # side by side: camera views are wider than they are tall
    (3, (2, 2)),
    (4, (2, 2)),
    (5, (2, 3)),
    (9, (3, 3)),
    (10, (3, 4)),
])
def test_the_grid_stays_close_to_square(count, expected):
    assert grid_shape(count) == expected


def test_every_camera_gets_a_tile_in_one_image():
    frames = {f"CAM-0{n}": frame() for n in range(1, 5)}
    wall = compose(frames, list(frames))

    assert wall is not None
    assert wall.ndim == 3
    # One image, not four - the whole point.
    rows, columns = grid_shape(4)
    assert wall.shape[0] % rows == 0
    assert wall.shape[1] % columns == 0


def test_the_wall_fits_on_a_screen():
    """Six 1080p cameras must not produce a 5760px window."""
    frames = {f"CAM-{n:02d}": frame(1920, 1080) for n in range(1, 7)}
    wall = compose(frames, list(frames), max_width=1600, max_height=900)

    assert wall.shape[1] <= 1600 + 160, f"wall is {wall.shape[1]} px wide"
    assert wall.shape[0] <= 900 + 120, f"wall is {wall.shape[0]} px tall"


# -------------------------------------------------------------- staying still


def test_a_camera_with_no_frame_yet_still_holds_its_place():
    """Otherwise every later camera renumbers the moment this one connects."""
    with_gap = compose({"CAM-01": frame(), "CAM-03": frame()},
                       ["CAM-01", "CAM-02", "CAM-03"])
    complete = compose({"CAM-01": frame(), "CAM-02": frame(), "CAM-03": frame()},
                       ["CAM-01", "CAM-02", "CAM-03"])

    assert with_gap.shape == complete.shape, "the layout moved when a camera joined"


def test_tile_order_does_not_depend_on_which_camera_spoke_last():
    """Frames arrive in whatever order threads finish; the wall must not care."""
    a = compose({"CAM-01": frame(value=40), "CAM-02": frame(value=200)},
                ["CAM-01", "CAM-02"])
    b = compose({"CAM-02": frame(value=200), "CAM-01": frame(value=40)},
                ["CAM-01", "CAM-02"])

    assert np.array_equal(a, b), "the tiles swapped places between frames"


def test_the_layout_is_decided_by_the_fleet_not_by_arrivals():
    """Three configured cameras is a 2x2 wall even when only one has a frame."""
    one_arrived = compose({"CAM-01": frame()}, ["CAM-01", "CAM-02", "CAM-03"])
    rows, columns = grid_shape(3)
    assert one_arrived.shape[0] // rows == one_arrived.shape[0] / rows
    assert one_arrived.shape[1] // columns == one_arrived.shape[1] / columns


# ------------------------------------------------------------ what tiles look like


def test_a_frame_is_letterboxed_rather_than_stretched():
    """A squashed frame is one the operator has to mentally un-squash."""
    tall = np.zeros((400, 100, 3), np.uint8)
    tall[:] = (0, 0, 255)
    wall = compose({"CAM-01": tall, "CAM-02": frame()}, ["CAM-01", "CAM-02"])

    cell = wall[:, :wall.shape[1] // 2]
    # Padding on the sides of a very tall frame means the aspect was kept.
    left_column = cell[cell.shape[0] // 2, :5]
    assert left_column.max() < 60, "the tall frame was stretched to fill the cell"


def test_cameras_of_different_resolutions_share_one_wall():
    wall = compose(
        {"CAM-01": frame(1920, 1080), "CAM-02": frame(320, 240),
         "CAM-03": frame(640, 480)},
        ["CAM-01", "CAM-02", "CAM-03"],
    )
    assert wall is not None
    assert wall.dtype == np.uint8


def test_an_empty_fleet_draws_nothing():
    assert compose({}, []) is None
    assert compose({}) is None


def test_a_zero_sized_frame_does_not_crash_the_wall():
    """A camera can hand over a broken frame; the other three must keep drawing."""
    wall = compose({"CAM-01": np.zeros((0, 0, 3), np.uint8), "CAM-02": frame()},
                   ["CAM-01", "CAM-02"])
    assert wall is not None


# ----------------------------------------------------------------- the runner


def test_the_fleet_titles_one_window_for_many_cameras(tmp_path):
    from app.config import SurveillanceConfig
    from app.runner import CameraSpec, MultiCameraRunner

    config = SurveillanceConfig(
        source="synthetic", detector="sim", synthetic_frames=5, view=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, color=False, quiet=True, print_summary=False,
    )
    many = MultiCameraRunner(
        [CameraSpec(camera_id="CAM-01", source="synthetic"),
         CameraSpec(camera_id="CAM-02", source="synthetic")], config)
    one = MultiCameraRunner([CameraSpec(camera_id="CAM-01", source="synthetic")], config)

    assert "2 cameras" in many._window_title()
    assert "CAM-01" in one._window_title()
