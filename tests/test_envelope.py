"""The detection envelope tool: how small a person can get before we lose them.

The tool ships as operational guidance for camera placement, so the parts that
shape its answer are worth pinning down - particularly the night simulation,
which is the half most likely to mislead if it quietly stopped degrading
anything.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "detection_envelope", Path("tools/detection_envelope.py")
)
envelope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(envelope)


def colourful(height=200, width=300):
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:, :, 0] = 200          # a strong blue cast
    frame[: height // 2, :, 2] = 180
    return frame


def test_the_night_simulation_removes_colour():
    """Infrared is monochrome; a simulation that keeps colour proves nothing."""
    out = envelope.night(colourful(), grain=0, blur=1)
    assert np.allclose(out[:, :, 0], out[:, :, 1], atol=1)
    assert np.allclose(out[:, :, 1], out[:, :, 2], atol=1)


def test_the_night_simulation_adds_grain():
    flat = np.full((120, 160, 3), 120, np.uint8)
    quiet = envelope.night(flat, grain=0, blur=1)
    noisy = envelope.night(flat, grain=25, blur=1)
    assert noisy.std() > quiet.std() * 3


def test_the_night_simulation_blurs_horizontally():
    """Motion blur along the direction of travel, not a general softening."""
    frame = np.zeros((80, 160, 3), np.uint8)
    frame[:, 78:82] = 255                       # a vertical bar
    out = envelope.night(frame, grain=0, blur=9)
    # The bar smears sideways, so its column spread widens.
    assert (out[40, :, 0] > 20).sum() > 4


def test_the_night_simulation_preserves_shape():
    frame = colourful(180, 240)
    assert envelope.night(frame).shape == frame.shape


@pytest.mark.parametrize("height", [400, 220, 120, 40])
def test_composing_scales_the_person_to_the_asked_height(height):
    background = np.zeros((envelope.FRAME_H, envelope.FRAME_W, 3), np.uint8)
    person = np.full((550, 210, 3), 200, np.uint8)
    frame = envelope.compose(background, person, height)

    assert frame.shape == (envelope.FRAME_H, envelope.FRAME_W, 3)
    lit = np.where(frame.max(axis=(1, 2)) > 100)[0]
    assert len(lit) == pytest.approx(height, abs=2)


def test_the_sweep_runs_from_large_to_small():
    """A floor only means something if the sweep passes through it."""
    assert list(envelope.HEIGHTS) == sorted(envelope.HEIGHTS, reverse=True)
    assert envelope.HEIGHTS[0] >= 300
    assert envelope.HEIGHTS[-1] <= 50
