"""Zone geometry is the safety-critical part of the MVP - test it hard."""

import pytest

from app.zones.geometry import (
    denormalize,
    normalize,
    point_in_polygon,
    validate_polygon,
)

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
# Concave "L" shape - the classic ray-casting failure case.
L_SHAPE = [(0.0, 0.0), (0.6, 0.0), (0.6, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0)]


@pytest.mark.parametrize("point", [(0.5, 0.5), (0.01, 0.99), (0.999, 0.001)])
def test_points_inside_square(point):
    assert point_in_polygon(point, SQUARE)


@pytest.mark.parametrize("point", [(1.5, 0.5), (-0.1, 0.5), (0.5, 2.0), (2.0, 2.0)])
def test_points_outside_square(point):
    assert not point_in_polygon(point, SQUARE)


@pytest.mark.parametrize("point", [(0.0, 0.5), (0.5, 0.0), (1.0, 1.0), (0.0, 0.0)])
def test_boundary_counts_as_inside(point):
    """A person standing on the fence line must not be silently ignored."""
    assert point_in_polygon(point, SQUARE)


def test_concave_polygon_notch_is_outside():
    assert point_in_polygon((0.3, 0.3), L_SHAPE)
    assert not point_in_polygon((0.8, 0.3), L_SHAPE)  # inside the notch
    assert point_in_polygon((0.8, 0.8), L_SHAPE)


def test_degenerate_polygon_is_never_inside():
    assert not point_in_polygon((0.5, 0.5), [(0.0, 0.0), (1.0, 1.0)])


def test_normalize_round_trip():
    assert normalize((480, 270), 960, 540) == (0.5, 0.5)
    assert denormalize((0.5, 0.5), 960, 540) == (480, 270)


def test_normalize_rejects_zero_size_frame():
    with pytest.raises(ValueError):
        normalize((10, 10), 0, 540)


def test_validate_polygon_rejects_pixel_coordinates():
    with pytest.raises(ValueError, match="normalized"):
        validate_polygon([[0, 0], [960, 0], [960, 540]])


def test_validate_polygon_requires_three_points():
    with pytest.raises(ValueError, match="at least 3"):
        validate_polygon([[0.0, 0.0], [1.0, 1.0]])
