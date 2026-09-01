"""Appearance-based re-identification.

Plain IoU tracking loses a person the moment they are occluded for longer than
`max_missed`, and hands back a fresh id. That silently resets dwell, so someone
who steps behind a pillar and returns stops being a loiterer. These tests pin
down that a track survives the occlusion, and that two different people are
still told apart.
"""

import numpy as np
import pytest

from app.detection.appearance import (
    blend,
    describe,
    similarity,
    torso_box,
)
from app.detection.base import Detection
from app.detection.tracker import IoUTracker

RED = (40, 40, 200)
BLUE = (200, 60, 40)
GREEN = (40, 180, 60)
DARK = (18, 18, 18)


def scene(colour, x, width=80, height=200, top=40):
    """A frame with one coloured person-sized block, plus its detection."""
    frame = np.full((300, 640, 3), 30, np.uint8)
    frame[top:top + height, x:x + width] = colour
    return frame, Detection("PERSON", 0.9, (x, top, x + width, top + height))


def blank():
    return np.full((300, 640, 3), 30, np.uint8)


# ------------------------------------------------------------- descriptors


def test_torso_box_sits_inside_the_person_box():
    inner = torso_box((100, 0, 200, 400))
    assert 100 < inner[0] < inner[2] < 200
    assert 0 < inner[1] < inner[3] < 400


def test_the_same_clothing_scores_near_one():
    a, _ = scene(RED, 100)
    b, _ = scene((45, 45, 210), 300)  # same shirt, slightly different light
    assert similarity(describe(a, (100, 40, 180, 240)), describe(b, (300, 40, 380, 240))) > 0.8


def test_different_clothing_scores_low():
    red, _ = scene(RED, 100)
    blue, _ = scene(BLUE, 100)
    assert similarity(describe(red, (100, 40, 180, 240)),
                      describe(blue, (100, 40, 180, 240))) < 0.3


def test_similarity_is_bounded():
    frame, det = scene(GREEN, 100)
    descriptor = describe(frame, det.bbox)
    assert 0.0 <= similarity(descriptor, descriptor) <= 1.0
    assert similarity(descriptor, descriptor) == pytest.approx(1.0, abs=1e-5)


def test_a_missing_descriptor_never_matches():
    frame, det = scene(RED, 100)
    assert similarity(describe(frame, det.bbox), None) == 0.0
    assert similarity(None, None) == 0.0


def test_a_box_with_no_colour_yields_no_descriptor():
    """Dark, washed-out pixels have meaningless hue and must not be described."""
    frame = np.full((300, 640, 3), 10, np.uint8)
    assert describe(frame, (100, 40, 180, 240)) is None


def test_a_tiny_box_yields_no_descriptor():
    frame, _ = scene(RED, 100)
    assert describe(frame, (100, 40, 103, 43)) is None


def test_blending_moves_toward_the_new_observation():
    red, _ = scene(RED, 100)
    blue, _ = scene(BLUE, 100)
    a = describe(red, (100, 40, 180, 240))
    b = describe(blue, (100, 40, 180, 240))

    merged = blend(a, b, weight=0.5)
    assert similarity(merged, b) > similarity(a, b)
    assert merged.sum() == pytest.approx(1.0, abs=1e-5)


def test_blending_handles_missing_sides():
    frame, det = scene(RED, 100)
    descriptor = describe(frame, det.bbox)
    assert blend(None, descriptor) is descriptor
    assert blend(descriptor, None) is descriptor


# ---------------------------------------------------------------- tracking


def walk(tracker, colour, positions, step=20):
    ids = []
    for x in positions:
        frame, det = scene(colour, x)
        ids.append(tracker.update([det], frame)[0].track_id)
    return ids


def occlude(tracker, frames):
    for _ in range(frames):
        tracker.update([], blank())


def test_identity_survives_a_long_occlusion():
    tracker = IoUTracker(max_missed=15)
    before = walk(tracker, RED, range(100, 200, 20))
    occlude(tracker, 40)
    after = walk(tracker, RED, range(400, 460, 20))

    assert set(before) == set(after) == {1}
    assert tracker.reids == 1


def test_a_different_person_gets_a_new_id():
    tracker = IoUTracker(max_missed=15)
    walk(tracker, RED, range(100, 200, 20))
    occlude(tracker, 40)
    after = walk(tracker, BLUE, range(400, 460, 20))

    assert set(after) == {2}
    assert tracker.reids == 0


def test_reid_can_be_switched_off():
    tracker = IoUTracker(max_missed=15, reid_enabled=False)
    walk(tracker, RED, range(100, 200, 20))
    occlude(tracker, 40)
    after = walk(tracker, RED, range(400, 460, 20))

    assert set(after) == {2}
    assert tracker.reids == 0


def test_a_track_is_forgotten_once_memory_expires():
    tracker = IoUTracker(max_missed=5, reid_memory=20)
    walk(tracker, RED, range(100, 200, 20))
    occlude(tracker, 60)
    after = walk(tracker, RED, range(400, 460, 20))

    assert set(after) == {2}
    assert tracker.lost_count == 0


def test_short_gaps_are_still_handled_by_iou_alone():
    """Nothing changes for the common case; re-ID is only for real occlusions."""
    tracker = IoUTracker(max_missed=15)
    walk(tracker, RED, [100, 120])
    occlude(tracker, 3)
    after = walk(tracker, RED, [128])

    assert after == [1]
    assert tracker.reids == 0


def test_a_visible_person_is_never_stolen_by_reid():
    """Two people on screen at once must keep their own ids."""
    tracker = IoUTracker()
    frame = np.full((300, 640, 3), 30, np.uint8)
    frame[40:240, 100:180] = RED
    frame[40:240, 400:480] = BLUE
    first = tracker.update(
        [Detection("PERSON", 0.9, (100, 40, 180, 240)),
         Detection("PERSON", 0.9, (400, 40, 480, 240))],
        frame,
    )
    second = tracker.update(
        [Detection("PERSON", 0.9, (108, 40, 188, 240)),
         Detection("PERSON", 0.9, (408, 40, 488, 240))],
        frame,
    )
    assert [d.track_id for d in first] == [d.track_id for d in second]
    assert tracker.reids == 0


def test_tracking_still_works_without_a_frame():
    """Callers that cannot supply pixels fall back to plain IoU."""
    tracker = IoUTracker()
    a = tracker.update([Detection("PERSON", 0.9, (100, 40, 180, 240))])
    b = tracker.update([Detection("PERSON", 0.9, (108, 40, 188, 240))])
    assert a[0].track_id == b[0].track_id == 1


def test_reset_clears_the_gallery():
    tracker = IoUTracker(max_missed=2)
    walk(tracker, RED, [100, 120])
    occlude(tracker, 5)
    assert tracker.lost_count == 1

    tracker.reset()
    assert tracker.lost_count == 0
    assert tracker.reids == 0
    assert walk(tracker, RED, [100]) == [1]


# ------------------------------------------------------- why it matters


def test_dwell_survives_an_occlusion_thanks_to_reid():
    """The payoff: a loiterer who steps out of sight is still a loiterer."""
    from app.behaviour.engine import LOITERING, BehaviourEngine
    from app.context.engine import ContextEngine
    from app.context.scene import SceneContext
    from app.zones.manager import ZoneManager

    zones = ZoneManager.from_dict({
        "zones": [{
            "name": "RESTRICTED", "kind": "RESTRICTED", "base_risk": 60,
            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        }]
    })
    night = SceneContext.build("CAM-01", force_night=True)
    tracker = IoUTracker(max_missed=5)
    context = ContextEngine(zones)
    behaviour = BehaviourEngine()

    def observe(track_id, at):
        position = (0.5, 0.9)
        return behaviour.classify(
            context.observe(track_id, position, zones.locate(position), night, at)
        )

    # Twelve seconds of standing still inside the zone.
    at = 0.0
    for x in (100, 102, 104, 106):
        frame, det = scene(RED, x)
        track_id = tracker.update([det], frame)[0].track_id
        found = observe(track_id, at)
        at += 4.0
    assert LOITERING in {b.name for b in found}

    # Step out of sight, then return.
    occlude(tracker, 20)
    at += 5.0
    frame, det = scene(RED, 300)
    track_id = tracker.update([det], frame)[0].track_id

    assert track_id == 1, "re-ID should have recovered the original track"
    assert LOITERING in {b.name for b in observe(track_id, at)}
