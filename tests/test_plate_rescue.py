"""Looking again at one vehicle, enlarged, when the frame pass found nothing.

The plate model runs once on the whole frame, which is cheap and works on
footage where plates are large. On real footage they often are not: measured on
the Delhi clip, the frame pass found plates on 36 of 171 vehicles, and looking
again at each missed vehicle - cropped and magnified three times - found 52.

That figure was first reported as 75, which was wrong, and wrong in an
instructive way: the extra 23 were a video watermark across a bus and a TATA
badge. Counting detections without looking at them is not measuring recall. The
geometry checks at the end of this file are what the correction produced.

The other dangerous part is the arithmetic. A
plate found inside a magnified crop is in the crop's coordinates, and it has to
be divided back down and shifted back out to the frame before anything uses it.
Get that wrong and the system files a picture of somebody else's bumper as
evidence of a plate, confidently and silently. So most of this is coordinates.

A stand-in model is used rather than the real weights: it returns boxes we
choose, which is the only way to check that a *known* box comes back at a
*known* place.
"""

import numpy as np
import pytest

from app.anpr.detector import PlateBox, PlateDetector


class FakeBoxes:
    """The shape ultralytics hands back, with no ultralytics."""

    def __init__(self, boxes, confs):
        self.xyxy = _Tensor(np.array(boxes, dtype=np.float32))
        self.conf = _Tensor(np.array(confs, dtype=np.float32))

    def __len__(self):
        return len(self.xyxy.value)


class _Tensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Returns the boxes it was told to, and records what it was shown."""

    def __init__(self, boxes, confs=None):
        self.boxes, self.confs = boxes, confs or [0.9] * len(boxes)
        self.seen = []

    def predict(self, source, **kwargs):
        self.seen.append(source)
        if not self.boxes:
            return [_Result(None)]
        return [_Result(FakeBoxes(self.boxes, self.confs))]


def detector(boxes, **kwargs):
    """A PlateDetector wired to a stand-in model."""
    found = PlateDetector.__new__(PlateDetector)
    found.__dict__.update(
        weights="fake", confidence=0.3, imgsz=960, device="cpu", min_width=60,
        rescue_crops=True, rescue_scale=3.0, rescue_min_width=12,
        model=FakeModel(boxes), available=True, last_error=None,
        _cache_key=None, _cache=[], passes=0, rescues=0, rescued=0,
    )
    for key, value in kwargs.items():
        setattr(found, key, value)
    return found


def frame(width=848, height=478):
    return np.full((height, width, 3), 60, np.uint8)


# ---------------------------------------------------------------- arithmetic


def test_a_plate_found_in_the_enlarged_crop_lands_back_in_the_frame():
    """The whole point. A box in crop pixels must come back in frame pixels."""
    # The vehicle sits at x 300..500, y 200..360. Inside its magnified crop
    # (3x), a plate at 150..270 maps back to 50..90 within the vehicle, which
    # is 350..390 in the frame.
    found = detector([[150, 300, 270, 330]])
    plates = found._rescue(frame(), (300, 200, 500, 360))

    assert len(plates) == 1
    assert plates[0].bbox == (350, 300, 390, 310)


def test_the_plate_stays_inside_the_vehicle_that_produced_it():
    """A plate mapped outside its own vehicle is arithmetic gone wrong."""
    vehicle = (300, 200, 500, 360)
    found = detector([[30, 60, 300, 120]])
    plate = found._rescue(frame(), vehicle)[0]

    vx1, vy1, vx2, vy2 = vehicle
    px1, py1, px2, py2 = plate.bbox
    assert vx1 <= px1 < px2 <= vx2, f"{plate.bbox} escaped {vehicle} horizontally"
    assert vy1 <= py1 < py2 <= vy2, f"{plate.bbox} escaped {vehicle} vertically"


def test_a_vehicle_at_the_origin_is_not_a_special_case():
    found = detector([[30, 30, 150, 60]])
    assert found._rescue(frame(), (0, 0, 200, 200))[0].bbox == (10, 10, 50, 20)


def test_the_magnification_is_undone_at_whatever_scale_is_set():
    """Scale is a knob; the arithmetic must follow it rather than assume 3."""
    for scale in (2.0, 3.0, 4.0):
        found = detector([[0, 0, 40 * scale, 10 * scale]], rescue_scale=scale)
        plate = found._rescue(frame(), (100, 100, 300, 260))[0]
        assert plate.bbox == (100, 100, 140, 110), f"scale {scale} mapped wrong"


def test_confidence_survives_the_round_trip():
    found = detector([[30, 30, 150, 60]])
    assert found._rescue(frame(), (0, 0, 200, 200))[0].confidence == pytest.approx(0.9)


def test_several_plates_come_back_most_confident_first():
    found = detector([[0, 0, 60, 20], [90, 0, 240, 30]])
    found.model.confs = [0.4, 0.8]
    plates = found._rescue(frame(), (100, 100, 400, 300))
    assert [round(p.confidence, 1) for p in plates] == [0.8, 0.4]


# ------------------------------------------------------------------ the gate


def test_the_rescue_floor_is_lower_than_the_frame_floor():
    """It is looking at an enlarged crop; the frame floor would reject everything.

    A 20 px plate magnified three times is 60 px in the crop. Judged by the
    frame's own 60 px floor it would survive by accident of magnification, which
    is why the rescue pass measures in original pixels and has its own floor.
    """
    found = detector([[0, 0, 60, 20]])          # 20 px once scaled back
    assert found.rescue_min_width < found.min_width
    assert found._rescue(frame(), (100, 100, 400, 300)), "20 px should pass a 12 px floor"

    strict = detector([[0, 0, 60, 20]], rescue_min_width=30)
    assert strict._rescue(frame(), (100, 100, 400, 300)) == []


def test_nothing_found_is_an_empty_list_not_an_error():
    assert detector([])._rescue(frame(), (100, 100, 400, 300)) == []


# -------------------------------------------------------- when it runs at all


def test_the_rescue_only_runs_when_the_frame_pass_found_nothing():
    """It costs a second forward pass, so it must not run when it is not needed."""
    # A plate the frame pass finds, inside the vehicle.
    found = detector([[350, 300, 450, 330]])
    plates = found.detect_on(frame(), (300, 200, 500, 360), cache_key=1)

    assert len(plates) == 1
    assert found.rescues == 0, "the frame pass succeeded; nothing to rescue"


def test_the_rescue_runs_when_the_frame_pass_missed():
    # The frame pass finds a plate, but on a different vehicle.
    found = detector([[50, 50, 150, 80]])
    found.detect_on(frame(), (300, 200, 500, 360), cache_key=1)
    assert found.rescues == 1


def test_the_rescue_can_be_switched_off():
    found = detector([[50, 50, 150, 80]], rescue_crops=False)
    assert found.detect_on(frame(), (300, 200, 500, 360), cache_key=1) == []
    assert found.rescues == 0


def test_a_detector_with_no_model_rescues_nothing():
    found = detector([])
    found.available = False
    assert found._rescue(frame(), (100, 100, 400, 300)) == []


def test_a_model_that_raises_is_reported_not_swallowed():
    """A broken rescue that silently returns nothing looks like 'no plates here'."""
    class Broken:
        def predict(self, **kwargs):
            raise RuntimeError("out of memory")

    found = detector([])
    found.model = Broken()
    assert found._rescue(frame(), (100, 100, 400, 300)) == []
    assert "out of memory" in found.last_error


# ------------------------------------------------------------------ the crop


def test_a_vehicle_box_running_off_the_frame_is_clamped():
    """A tracker box can extend past the edge; slicing with it must not fail."""
    found = detector([[0, 0, 60, 20]])
    plates = found._rescue(frame(200, 200), (150, 150, 400, 400))
    assert plates, "a partly visible vehicle should still be searched"
    assert all(p.bbox[2] <= 400 for p in plates)


def test_a_vehicle_too_small_to_crop_is_skipped():
    assert detector([[0, 0, 60, 20]])._rescue(frame(), (100, 100, 104, 104)) == []


def test_an_empty_crop_is_skipped():
    assert detector([[0, 0, 60, 20]])._rescue(frame(), (500, 500, 400, 400)) == []


def test_the_model_is_shown_a_magnified_crop_not_the_whole_frame():
    """If it were shown the frame, the rescue would be the pass that just failed."""
    found = detector([[0, 0, 60, 20]])
    found._rescue(frame(848, 478), (300, 200, 500, 360))

    shown = found.model.seen[-1]
    assert shown.shape[:2] == (160 * 3, 200 * 3), f"got {shown.shape[:2]}"


def test_the_counters_record_what_the_rescue_did():
    found = detector([[0, 0, 60, 20], [90, 0, 150, 30]])
    found._rescue(frame(), (100, 100, 400, 300))
    assert found.rescues == 1
    assert found.rescued == 2


# ------------------------------------------------------------------ geometry


def test_a_box_covering_most_of_the_vehicle_is_not_a_plate():
    """This is what a watermark across a bus looked like, and it was counted.

    The first version of this rescue had no geometry check and reported 43.9%
    of vehicles yielding a plate. Inspecting what it found showed a video
    watermark and a TATA badge among them; with the check the honest figure is
    30.4%. The lesson is in the test rather than only in a comment: a recall
    number that was never checked for what it found is not a recall number.
    """
    vehicle = (0, 0, 400, 300)
    found = detector([[0, 0, 380 * 3, 90 * 3]])          # 380 px of a 400 px vehicle
    assert found._rescue(frame(), vehicle) == []


def test_a_plate_sized_box_on_the_same_vehicle_is_kept():
    vehicle = (0, 0, 400, 300)
    found = detector([[0, 0, 120 * 3, 30 * 3]])          # 120 px, plate-like
    assert len(found._rescue(frame(), vehicle)) == 1


def test_a_box_taller_than_it_is_wide_is_not_a_plate():
    found = detector([[0, 0, 30 * 3, 60 * 3]])
    assert found._rescue(frame(), (0, 0, 400, 300)) == []


def test_a_hairline_box_is_not_a_plate_either():
    """Ten times wider than tall is a trim strip, not a number plate."""
    found = detector([[0, 0, 150 * 3, 12 * 3]])
    assert found._rescue(frame(), (0, 0, 400, 300)) == []


def test_both_shapes_of_indian_plate_survive_the_check():
    """Single-line is about 4.7:1 and two-line about 2:1; both are real."""
    vehicle = (0, 0, 400, 300)
    single = detector([[0, 0, 141 * 3, 30 * 3]])         # 4.7:1
    two_line = detector([[0, 0, 100 * 3, 50 * 3]])       # 2:1
    assert single._rescue(frame(), vehicle), "a single-line plate was rejected"
    assert two_line._rescue(frame(), vehicle), "a two-line plate was rejected"


def test_the_geometry_limits_are_adjustable():
    vehicle = (0, 0, 400, 300)
    wide = detector([[0, 0, 380 * 3, 90 * 3]], rescue_max_vehicle_fraction=1.0,
                    rescue_max_aspect=99.0)
    assert wide._rescue(frame(), vehicle), "the limits should be tunable, not baked in"


def test_a_zero_height_box_is_rejected_rather_than_dividing_by_zero():
    found = detector([[0, 0, 120 * 3, 0]])
    assert found._rescue(frame(), (0, 0, 400, 300)) == []
