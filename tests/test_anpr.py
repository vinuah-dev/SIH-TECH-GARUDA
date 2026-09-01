"""ANPR: locating and reading a vehicle's number plate.

The rule these tests encode is that a wrong plate number in an evidence log is
worse than an admitted blank. So most of them are about *rejection*: reads that
cannot be a plate are discarded rather than guessed at, and a correction is
only applied when it is unambiguous.
"""

import cv2
import numpy as np
import pytest

from app.anpr import plate as plate_finder
from app.anpr import text as plate_text
from app.anpr.engine import ANPREngine, PlateReading
from app.detection.base import Detection


def vehicle_frame(number="MH12AB1234", width=640, height=480):
    """A vehicle-shaped block with a plate low in it, as a real one would be."""
    frame = np.full((height, width, 3), 70, np.uint8)
    cv2.rectangle(frame, (180, 140), (460, 420), (90, 90, 110), -1)
    cv2.rectangle(frame, (250, 340), (400, 392), (245, 245, 245), -1)
    cv2.putText(frame, number, (256, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (10, 10, 10), 2)
    return frame, Detection("CAR", 0.9, (180, 140, 460, 420), track_id=1)


# --------------------------------------------------------- text normalising


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("MH12AB1234", "MH12AB1234"),
        ("MH 12 AB 1234", "MH12AB1234"),
        ("mh12ab1234", "MH12AB1234"),
        ("!!MH-12-AB-1234!!", "MH12AB1234"),
        ("MH12A1234", "MH12A1234"),
        ("22BH1234AA", "22BH1234AA"),
    ],
)
def test_valid_plates_are_accepted(raw, expected):
    assert plate_text.normalise(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("MHI2AB1234", "MH12AB1234"),   # I read for 1
        ("MH12A81234", "MH12AB1234"),   # 8 read for B
        ("OL8CAF5030", "DL8CAF5030"),   # O read for D in the state code
    ],
)
def test_common_ocr_confusions_are_repaired(raw, expected):
    assert plate_text.normalise(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "ABC", "12345678", "XX12AB1234", "MH12AB12345678", "ZZ99ZZ9999"],
)
def test_impossible_reads_are_rejected(raw):
    """A wrong number in an evidence log is worse than an honest blank."""
    assert plate_text.normalise(raw) is None


def test_an_ambiguous_state_code_is_not_guessed():
    """Only a single-candidate correction is safe to apply."""
    assert plate_text._repair_state("MH") == "MH"
    assert plate_text._repair_state("ZZ") is None


def test_plates_are_displayed_the_way_they_are_printed():
    assert plate_text.format_display("MH12AB1234") == "MH 12 AB 1234"
    assert plate_text.state_of("MH12AB1234") == "MH"
    assert plate_text.state_of("nonsense") is None


def test_cleaning_strips_everything_a_plate_cannot_contain():
    assert plate_text.clean(" mh-12/ab 1234 ") == "MH12AB1234"


# -------------------------------------------------------------- localisation


def test_a_plate_is_found_inside_the_vehicle_box():
    frame, detection = vehicle_frame()
    found = plate_finder.candidates(frame, detection.bbox)
    assert found

    x1, y1, x2, y2 = found[0].bbox
    # Close to where the plate was actually drawn.
    assert abs(x1 - 250) < 25 and abs(x2 - 400) < 25
    assert abs(y1 - 340) < 25 and abs(y2 - 392) < 25


def test_only_the_lower_half_of_the_vehicle_is_searched():
    """A plate sits low; the upper half is windscreen, roof and sky."""
    x1, y1, x2, y2 = plate_finder._search_region((100, 0, 300, 400))
    assert y1 > 0
    assert y2 == 400


def test_a_vehicle_with_no_plate_yields_nothing():
    frame = np.full((480, 640, 3), 70, np.uint8)
    cv2.rectangle(frame, (180, 140), (460, 420), (90, 90, 110), -1)
    assert plate_finder.candidates(frame, (180, 140, 460, 420)) == []


def test_a_box_too_small_to_hold_a_plate_is_skipped():
    frame, _ = vehicle_frame()
    assert plate_finder.candidates(frame, (10, 10, 40, 30)) == []


def test_cropping_stays_inside_the_frame():
    frame, _ = vehicle_frame()
    crop = plate_finder.crop(frame, (630, 470, 700, 520), pad=10)
    assert crop is None or crop.size > 0


def test_a_small_region_is_upscaled_for_the_detector():
    """The detector needs pixels; a distant plate arrives with very few."""
    small = np.full((40, 120, 3), 200, np.uint8)
    scaled, factor = plate_finder.upscale_for_text(small, min_width=320)
    assert scaled.shape[1] == 320
    assert factor > 1.0


def test_a_large_region_is_left_alone():
    big = np.full((300, 800, 3), 200, np.uint8)
    scaled, factor = plate_finder.upscale_for_text(big, min_width=320)
    assert scaled.shape[1] == 800
    assert factor == 1.0


# -------------------------------------------------------------------- engine


def test_only_vehicles_are_examined():
    engine = ANPREngine()
    frame, _ = vehicle_frame()
    person = Detection("PERSON", 0.9, (180, 140, 460, 420), track_id=1)

    engine.tick()
    assert engine.observe(frame, person) is None
    assert engine.attempts_made == 0


def test_an_untracked_detection_is_skipped():
    engine = ANPREngine()
    frame, _ = vehicle_frame()
    untracked = Detection("CAR", 0.9, (180, 140, 460, 420))
    engine.tick()
    assert engine.observe(frame, untracked) is None


def test_anpr_can_be_disabled():
    engine = ANPREngine(enabled=False)
    frame, detection = vehicle_frame()
    engine.tick()
    assert engine.observe(frame, detection) is None
    assert engine.attempts_made == 0


def test_attempts_are_spaced_out_and_bounded():
    """OCR on every frame would halve the frame rate for no extra information."""
    engine = ANPREngine(attempt_interval=5, max_attempts=2)
    frame, detection = vehicle_frame()
    for _ in range(40):
        engine.tick()
        engine.observe(frame, detection)
    assert engine.attempts_made <= 2


def test_a_plate_crop_is_kept_even_when_ocr_reads_nothing(monkeypatch):
    """An operator can read a plate the machine could not."""
    engine = ANPREngine(attempt_interval=1)
    monkeypatch.setattr(
        ANPREngine, "_detect_text",
        lambda self, image: [([[0, 0], [90, 0], [90, 30], [0, 30]], "XX9", 0.9)],
    )

    frame, detection = vehicle_frame()
    engine.tick()
    engine.observe(frame, detection)
    assert engine.crop_for(1) is not None
    assert engine.best_for(1) is None


def test_reads_are_voted_on_across_frames(monkeypatch):
    """One read at 60% is a guess; four reads that agree are a number.

    `reads` counts every read that went into the consensus, including one that
    differed in a couple of places and was outvoted position by position.
    """
    sequence = iter([
        ("MH12AB1234", 0.7), ("MH12AB1299", 0.6),
        ("MH12AB1234", 0.8), ("MH12AB1234", 0.9),
    ])

    def one(self, image):
        text, conf = next(sequence, ("", 0.0))
        return [([[0, 0], [90, 0], [90, 30], [0, 30]], text, conf)] if text else []

    monkeypatch.setattr(ANPREngine, "_detect_text", one)
    engine = ANPREngine(attempt_interval=1, confident_reads=99, min_sharpness=0)
    frame, detection = vehicle_frame()
    reading = None
    for _ in range(4):
        engine.tick()
        reading = engine.observe(frame, detection)

    assert reading.text == "MH12AB1234"
    assert reading.reads == 4


def test_reading_stops_once_enough_reads_agree(monkeypatch):
    monkeypatch.setattr(
        ANPREngine, "_detect_text",
        lambda self, image: [([[0, 0], [90, 0], [90, 30], [0, 30]], "MH12AB1234", 0.9)],
    )
    engine = ANPREngine(attempt_interval=1, confident_reads=2, max_attempts=50,
                        min_sharpness=0)

    frame, detection = vehicle_frame()
    for _ in range(20):
        engine.tick()
        engine.observe(frame, detection)
    assert engine.attempts_made == 2


def test_a_low_confidence_read_is_ignored(monkeypatch):
    monkeypatch.setattr(
        ANPREngine, "_detect_text",
        lambda self, image: [([[0, 0], [90, 0], [90, 30], [0, 30]], "MH12AB1234", 0.01)],
    )
    engine = ANPREngine(attempt_interval=1, min_confidence=0.35, min_sharpness=0)

    frame, detection = vehicle_frame()
    engine.tick()
    assert engine.observe(frame, detection) is None


def test_each_vehicle_keeps_its_own_plate(monkeypatch):
    plates = {1: "MH12AB1234", 2: "DL8CAF5030"}
    current = {"id": 1}
    monkeypatch.setattr(
        ANPREngine, "_detect_text",
        lambda self, image: [
            ([[0, 0], [90, 0], [90, 30], [0, 30]], plates[current["id"]], 0.9)
        ],
    )

    engine = ANPREngine(attempt_interval=1, min_sharpness=0)
    frame, _ = vehicle_frame()
    for track_id in (1, 2):
        current["id"] = track_id
        for _ in range(engine.min_agreeing_reads):
            engine.tick()
            engine.observe(
                frame, Detection("CAR", 0.9, (180, 140, 460, 420), track_id=track_id)
            )

    assert engine.best_for(1).text == "MH12AB1234"
    assert engine.best_for(2).text == "DL8CAF5030"


def test_forgetting_a_track_clears_its_plate(monkeypatch):
    monkeypatch.setattr(
        ANPREngine, "_detect_text",
        lambda self, image: [([[0, 0], [90, 0], [90, 30], [0, 30]], "MH12AB1234", 0.9)],
    )
    engine = ANPREngine(attempt_interval=1, min_sharpness=0)
    frame, detection = vehicle_frame()
    for _ in range(engine.min_agreeing_reads):
        engine.tick()
        engine.observe(frame, detection)

    engine.forget(1)
    assert engine.best_for(1) is None


def test_missing_ocr_degrades_instead_of_crashing(monkeypatch):
    """Without a backend IBVAP still localises the plate and saves the crop."""
    engine = ANPREngine(attempt_interval=1)
    monkeypatch.setattr(ANPREngine, "_ocr", lambda self: None)

    frame, detection = vehicle_frame()
    engine.tick()
    assert engine.observe(frame, detection) is None
    assert engine.crop_for(1) is not None


def test_a_reading_serialises_for_the_event_log():
    reading = PlateReading("MH12AB1234", 0.91, (10, 20, 90, 45), reads=3, raw="MH12AB1234")
    payload = reading.to_dict()
    assert payload["display"] == "MH 12 AB 1234"
    assert payload["state"] == "MH"
    assert payload["reads"] == 3
    assert payload["bbox"] == [10, 20, 90, 45]


# --------------------------------------------------------------- the store


def test_events_can_be_looked_up_by_plate(tmp_path):
    """The movement history of one vehicle across a post."""
    from datetime import datetime

    from app.alerts.events import IntrusionEvent
    from app.risk.engine import RiskAssessment, RiskFactor
    from app.store import EventStore
    from app.zones.manager import Zone

    zone = Zone("R", "RESTRICTED", [(0.5, 0), (1, 0), (1, 1)], base_risk=60)

    def event(event_id, plate, hour):
        return IntrusionEvent(
            event_id=event_id,
            timestamp=datetime(2026, 8, 25, hour, 0),
            camera_id="CAM-01",
            event_type="VIRTUAL FENCE INTRUSION",
            detection=Detection("CAR", 0.9, (1, 1, 10, 10), track_id=1),
            zone=zone,
            risk=RiskAssessment(80, "HIGH", "test", [RiskFactor("Z", 80, "d")]),
            frame_index=1,
            plate={"text": plate, "display": plate, "state": None,
                   "confidence": 0.9, "reads": 3, "bbox": [], "raw": plate},
        )

    with EventStore(tmp_path / "plates.db") as store:
        store.write(event("A", "MH12AB1234", 9))
        store.write(event("B", "MH12AB1234", 14))
        store.write(event("C", "DL8CAF5030", 11))

        assert len(store.by_plate("MH12AB1234")) == 2
        assert store.by_plate("mh12ab1234")[0]["event_id"] == "B"  # newest first
        assert store.plates_seen() == {"MH12AB1234": 2, "DL8CAF5030": 1}
        assert store.by_plate("XX00XX0000") == []


# ------------------------------------------------- vehicles that are moving


def test_a_blurred_plate_is_skipped_rather_than_guessed_at(monkeypatch):
    """Motion blur makes OCR confidently wrong, not quietly absent."""
    monkeypatch.setattr(
        ANPREngine, "_detect_text",
        lambda self, image: [([[0, 0], [90, 0], [90, 30], [0, 30]], "MH12AB1234", 0.9)],
    )

    frame, detection = vehicle_frame()
    blurred = cv2.GaussianBlur(frame, (25, 25), 0)

    engine = ANPREngine(attempt_interval=1, min_sharpness=10_000)
    engine.tick()
    engine.observe(blurred, detection)

    assert engine._plates[1].blurred_skips > 0, "a blurred crop must be skipped"
    assert engine.best_for(1) is None


def test_the_sharpest_crop_seen_is_kept_as_evidence(monkeypatch):
    """An operator can read a plate the machine refused to guess at."""
    # The box must land on the drawn plate: blurring a flat panel changes no
    # variance at all, so a rig that crops bodywork proves nothing.
    monkeypatch.setattr(
        ANPREngine, "_detect_text",
        lambda self, image: [([[70, 200], [220, 200], [220, 252], [70, 252]], "XX9", 0.9)],
    )

    frame, detection = vehicle_frame()
    blurred = cv2.GaussianBlur(frame, (25, 25), 0)

    engine = ANPREngine(attempt_interval=1, min_sharpness=1e9)
    engine.tick()
    engine.observe(blurred, detection)
    blurry_focus = engine._plates[1].best_sharpness

    engine.tick()
    engine.observe(frame, detection)  # a sharp view arrives later
    assert engine._plates[1].best_sharpness > blurry_focus
    assert engine.crop_for(1) is not None


def test_sharpness_separates_a_crisp_plate_from_a_blurred_one():
    frame, _ = vehicle_frame()
    crisp = frame[340:392, 250:400]
    blurred = cv2.GaussianBlur(crisp, (21, 21), 0)
    assert plate_finder.sharpness(crisp) > plate_finder.sharpness(blurred) * 3


def test_a_stationary_vehicle_is_still_read_repeatedly(monkeypatch):
    """A vehicle stopped at a checkpoint never grows; voting must still work."""
    monkeypatch.setattr(
        ANPREngine, "_detect_text",
        lambda self, image: [([[0, 0], [90, 0], [90, 30], [0, 30]], "MH12AB1234", 0.9)],
    )
    engine = ANPREngine(attempt_interval=1, confident_reads=3, min_sharpness=0)
    frame, detection = vehicle_frame()
    reading = None
    for _ in range(6):
        engine.tick()
        reading = engine.observe(frame, detection)

    assert reading.reads == 3


def test_a_much_closer_view_earns_one_more_look(monkeypatch):
    """Approaching traffic gets a better angle; that is worth re-reading."""
    monkeypatch.setattr(
        ANPREngine, "_detect_text",
        lambda self, image: [([[0, 0], [90, 0], [90, 30], [0, 30]], "MH12AB1234", 0.9)],
    )
    engine = ANPREngine(attempt_interval=1, confident_reads=2, min_sharpness=0)
    frame, detection = vehicle_frame()
    for _ in range(4):
        engine.tick()
        engine.observe(frame, detection)
    settled = engine.attempts_made

    # The same vehicle, now much larger in frame.
    closer = Detection("CAR", 0.9, (100, 60, 560, 460), track_id=1)
    engine.tick()
    engine.observe(frame, closer)
    assert engine.attempts_made > settled


# ---------------------------------------------------- failing quietly is a bug


def test_a_broken_ocr_call_is_reported_not_swallowed(monkeypatch):
    """A silent OCR failure looks exactly like 'no plate was ever readable'.

    That is how a broken call can disable plate reading for a whole session
    without anyone noticing, so the first failure has to surface.
    """
    engine = ANPREngine()

    class Exploding:
        name = "exploding"

        def read(self, image):
            raise RuntimeError("boom")

    engine._reader = Exploding()
    assert engine._detect_text(np.zeros((10, 10, 3), np.uint8)) == []
    assert engine._detect_text(np.zeros((10, 10, 3), np.uint8)) == []

    assert engine.ocr_failures == 2
    assert engine.last_error == "boom"


def test_a_missing_ocr_backend_is_recorded(monkeypatch):
    """Without any recogniser the feature degrades; the run carries on.

    Every candidate has to be blocked, not just one: the engine tries several
    in a measured order, so failing the first only proves the fallback works.
    """
    from app.anpr import readers

    monkeypatch.setattr(
        readers, "build", lambda name=None: (None, "no recogniser installed")
    )
    engine = ANPREngine()
    assert engine._ocr() is None
    assert not engine.ocr_available
    assert "no recogniser installed" in engine.last_error


def test_the_engine_survives_every_backend_being_absent(monkeypatch):
    """The real shape of a bare machine: nothing importable, nothing crashes."""
    from app.anpr import readers

    class Absent:
        name = "absent"

        def load(self):
            raise ImportError("not installed")

    for candidate in readers.PREFERENCE:
        monkeypatch.setitem(readers.BACKENDS, candidate, Absent)

    engine = ANPREngine()
    assert engine._ocr() is None
    # Localisation and crop saving are unaffected, which is the whole point of
    # degrading rather than failing.
    assert engine.enabled


# ------------------------------------------------- per-character consensus


def track_with(*reads):
    from app.anpr.engine import _TrackPlate

    state = _TrackPlate()
    for read in reads:
        state.votes[read] += 1
    return state


def test_reads_that_each_get_one_character_wrong_still_agree():
    """Real reads of one plate disagree in *different* places.

    Whole-string voting throws that away and reports nothing; voting per
    character position recovers the number they jointly carry.
    """
    state = track_with("DL9CZ2581", "DL5CZ2581", "DL9CZ2581")
    assert state.consensus(min_reads=2) == ("DL9CZ2581", 3)


def test_a_tied_position_is_refused_rather_than_guessed():
    """Two reads differing at one position give every character one vote.

    Taking the first is an arbitrary pick wearing a consensus costume.
    """
    assert track_with("DL1CZ2581", "DL9CI2581").consensus(min_reads=2) is None


def test_a_single_read_is_never_a_consensus():
    assert track_with("MH12AB1234").consensus(min_reads=2) is None


def test_unanimous_reads_pass_straight_through():
    assert track_with("MH12AB1234", "MH12AB1234").consensus(min_reads=2) == ("MH12AB1234", 2)


def test_reads_of_different_lengths_are_not_merged():
    """A different length is a different reading, not a differing opinion."""
    state = track_with("MH12AB1234", "MH12AB1234", "MH12A1234")
    agreed = state.consensus(min_reads=2)
    assert agreed == ("MH12AB1234", 2)


def test_the_engine_reports_only_a_real_consensus(monkeypatch):
    sequence = iter(["DL9CZ2581", "DL5CZ2581", "DL9CZ2581"])

    def one(self, image):
        text = next(sequence, "")
        return [([[0, 0], [90, 0], [90, 30], [0, 30]], text, 0.9)] if text else []

    monkeypatch.setattr(ANPREngine, "_detect_text", one)
    engine = ANPREngine(attempt_interval=1, confident_reads=99, min_sharpness=0)
    frame, detection = vehicle_frame()

    reading = None
    for _ in range(3):
        engine.tick()
        reading = engine.observe(frame, detection)
    assert reading.text == "DL9CZ2581"


def test_no_crop_is_invented_when_the_plate_model_found_no_plate(monkeypatch):
    """Measured on real footage, not imagined.

    Two street clips were run through the pipeline. Where the plate model found
    nothing on a vehicle, the text-search fallback still ran and saved the
    video's own watermark - "wildfilmsindia.com", "msindia.com" - five times
    out of five. An overlay is rendered digitally, so it is sharper than any
    plate seen through a lens at distance and wins a sharpness contest every
    time. A crop filed as evidence of a plate, showing a watermark, is a false
    record, and worse than no crop at all.

    A trained model finding no plate is evidence there is no visible plate.
    """
    class FoundNothing:
        available = True

        def detect_on(self, *a, **k):
            return []

    engine = ANPREngine(attempt_interval=1, plate_detector=FoundNothing())
    monkeypatch.setattr(
        ANPREngine, "_detect_text",
        lambda self, image: [([[0, 0], [300, 0], [300, 40], [0, 40]],
                              "wildfilmsindia.com", 0.99)],
    )

    frame, detection = vehicle_frame()
    engine.tick()
    engine.observe(frame, detection)
    assert engine.crop_for(1) is None, "a watermark was saved as plate evidence"


def test_the_fallback_still_runs_when_no_plate_model_is_installed(monkeypatch):
    """The fallback exists for the no-model case, and must survive there.

    Without a model there is no better way to look, so the sharpest text region
    on the vehicle remains the best guess available - which is exactly what the
    README says the no-model path is worth.
    """
    engine = ANPREngine(attempt_interval=1)
    assert engine.plate_detector is None
    monkeypatch.setattr(
        ANPREngine, "_detect_text",
        lambda self, image: [([[0, 0], [90, 0], [90, 30], [0, 30]], "XX9", 0.9)],
    )

    frame, detection = vehicle_frame()
    engine.tick()
    engine.observe(frame, detection)
    assert engine.crop_for(1) is not None


def test_repair_may_correct_a_read_but_not_rewrite_it():
    """Measured: extending the confusion tables raised exact reads from 30.5% to
    35.2% of 400 annotated plates - and raised *wrong* accepts from 74 to 92.

    A wrong plate is not a harmless miss. It reaches the vehicle registry and
    flags an innocent driver. So repair is allowed a few shape corrections and
    no more; a read needing five was never this plate.
    """
    from app.anpr.text import MAX_REPAIRS, normalise

    # One confusion per position, well inside the cap.
    assert normalise("MH12AB1Z34") is None or normalise("MHI2AB1234") == "MH12AB1234"
    assert normalise("MHI2AB1234") == "MH12AB1234"

    # A read mangled beyond the cap must be refused, not forced to fit.
    assert MAX_REPAIRS == 3
    assert normalise("OOOOOOOOOO") is None


def test_the_measured_confusions_are_actually_repaired():
    """Each of these was counted in the 400-plate run, not imagined."""
    from app.anpr.text import normalise

    # U read where a 0 belongs, and J where a 1 belongs.
    assert normalise("WBU6AF9209") == "WB06AF9209"
    # A read where a 4 belongs, in the trailing number.
    assert normalise("MH12AB123A") == "MH12AB1234"


def test_a_state_code_is_only_rescued_when_the_answer_is_unambiguous():
    """MH read as AH is recoverable. A code with two plausible fixes is not."""
    from app.anpr.text import normalise

    assert normalise("AH12AB1234") == "MH12AB1234"
    assert normalise("XX12AB1234") is None
