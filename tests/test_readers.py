"""Choosing which recogniser reads a plate crop.

Three were measured against 400 annotated Indian plates, each cropped to its
annotated box so only recognition was judged:

    EasyOCR                     34.5% exact, 71.2% chars, 0.56s a plate
    PaddleOCR, full pipeline    60.2% exact, 80.3% chars, 5.59s a plate
    PaddleOCR, recogniser only  73.2% exact, 90.7% chars, 0.22s a plate

The last is both the most accurate and the fastest, because the plate is
already located by the time any of it runs - so PaddleOCR's own text detection
is paying to find what has been found, and fragmenting the plate while it does.

These tests are about the selection and the contract, not the models: they run
on a machine with neither installed, because a recogniser missing must degrade
plate reading rather than stop the run.
"""

import numpy as np
import pytest

from app.anpr import readers


def crop(width=140, height=40):
    return np.full((height, width, 3), 200, np.uint8)


class Loads:
    """A backend that loads and returns what it is told to."""

    name = "stand-in"

    def __init__(self, results=()):
        self.results = results
        self.loaded = False

    def load(self):
        self.loaded = True

    def read(self, image):
        return self.results


class Fails:
    name = "broken"

    def load(self):
        raise RuntimeError("no such module")

    def read(self, image):
        raise AssertionError("must never be reached")


# ------------------------------------------------------------------ selection


def test_the_preferred_backend_is_the_measured_best_one():
    """paddle-rec is first because it measured best, not because it is newest."""
    assert readers.PREFERENCE[0] == "paddle-rec"
    assert "easyocr" in readers.PREFERENCE, "a fallback must exist"


def test_a_named_backend_is_used_even_if_another_would_be_preferred(monkeypatch):
    chosen = Loads()
    monkeypatch.setitem(readers.BACKENDS, "easyocr", lambda: chosen)

    reader, problem = readers.build("easyocr")
    assert reader is chosen and problem is None
    assert chosen.loaded


def test_an_unknown_backend_is_named_in_the_error():
    reader, problem = readers.build("tesseract")
    assert reader is None
    assert "tesseract" in problem
    assert "paddle-rec" in problem, "the error should list what is available"


def test_a_named_backend_that_will_not_load_says_why(monkeypatch):
    monkeypatch.setitem(readers.BACKENDS, "paddle", Fails)
    reader, problem = readers.build("paddle")
    assert reader is None
    assert "no such module" in problem


def test_a_named_backend_does_not_quietly_fall_back(monkeypatch):
    """Asking for one recogniser and silently getting another invalidates a run."""
    monkeypatch.setitem(readers.BACKENDS, "paddle", Fails)
    monkeypatch.setitem(readers.BACKENDS, "easyocr", lambda: Loads())

    reader, problem = readers.build("paddle")
    assert reader is None, "it fell back to a recogniser nobody asked for"


def test_with_no_name_the_first_backend_that_loads_wins(monkeypatch):
    wanted = Loads()
    monkeypatch.setitem(readers.BACKENDS, "paddle-rec", lambda: wanted)
    monkeypatch.setitem(readers.BACKENDS, "easyocr", Fails)

    reader, problem = readers.build()
    assert reader is wanted and problem is None


def test_the_fallback_is_used_when_the_preferred_one_is_absent(monkeypatch):
    """A machine without PaddleOCR must still read plates, not silently read none."""
    fallback = Loads()
    monkeypatch.setitem(readers.BACKENDS, "paddle-rec", Fails)
    monkeypatch.setitem(readers.BACKENDS, "easyocr", lambda: fallback)

    reader, problem = readers.build()
    assert reader is fallback and problem is None


def test_when_nothing_loads_every_reason_is_reported(monkeypatch):
    for candidate in readers.PREFERENCE:
        monkeypatch.setitem(readers.BACKENDS, candidate, Fails)

    reader, problem = readers.build()
    assert reader is None
    for candidate in readers.PREFERENCE:
        assert candidate in problem, f"{candidate} failed silently"


# ------------------------------------------------------------------- contract


def test_every_backend_returns_box_text_confidence(monkeypatch):
    """Plate selection uses *where* a string was found, not only what it said."""
    reader = readers.PaddleRecogniser()

    class Stub:
        def predict(self, image):
            return [{"rec_text": "MH12AB1234", "rec_score": 0.83}]

    reader._reader = Stub()
    results = reader.read(crop(140, 40))

    assert len(results) == 1
    box, text, confidence = results[0]
    assert text == "MH12AB1234"
    assert confidence == pytest.approx(0.83)
    assert box == [[0.0, 0.0], [140.0, 0.0], [140.0, 40.0], [0.0, 40.0]], (
        "with detection skipped, the box is the whole crop"
    )


def test_the_recogniser_only_backend_drops_empty_reads():
    reader = readers.PaddleRecogniser()

    class Stub:
        def predict(self, image):
            return [{"rec_text": "", "rec_score": 0.9},
                    {"rec_text": "MH12AB1234", "rec_score": 0.7},
                    {"nothing": "useful"}]

    reader._reader = Stub()
    assert [t for _, t, _ in reader.read(crop())] == ["MH12AB1234"]


def test_a_backend_with_no_results_returns_an_empty_list():
    reader = readers.PaddleRecogniser()

    class Stub:
        def predict(self, image):
            return None

    reader._reader = Stub()
    assert reader.read(crop()) == []


def test_the_full_paddle_backend_keeps_the_box_each_string_came_from():
    """It is kept precisely because it can separate the rows of a two-line plate."""
    reader = readers.PaddleOCRReader()

    class Stub:
        def predict(self, image):
            return [{
                "rec_texts": ["MH12", "AB1234"],
                "rec_scores": [0.8, 0.9],
                "rec_polys": [[[0, 0], [60, 0], [60, 20], [0, 20]],
                              [[0, 22], [80, 22], [80, 42], [0, 42]]],
            }]

    reader._reader = Stub()
    results = reader.read(crop())

    assert [t for _, t, _ in results] == ["MH12", "AB1234"]
    # The second row sits below the first, which is how it can be reassembled.
    assert results[0][0][0][1] < results[1][0][0][1]


# --------------------------------------------------------------- in the engine


def test_the_engine_reports_which_recogniser_it_loaded(monkeypatch):
    from app.anpr.engine import ANPREngine

    class Named(Loads):
        name = "stand-in recogniser"

    monkeypatch.setattr(readers, "build", lambda name=None: (Named(), None))
    engine = ANPREngine()
    engine._ocr()
    assert engine.reader_name == "stand-in recogniser"


def test_a_missing_recogniser_warns_once_and_keeps_going(monkeypatch):
    """Plates are still located and their crops still saved; an operator can read one."""
    from app import ui
    from app.anpr.engine import ANPREngine

    said = []
    monkeypatch.setattr(ui, "warn", said.append)
    monkeypatch.setattr(readers, "build", lambda name=None: (None, "nothing installed"))

    engine = ANPREngine()
    for _ in range(4):
        assert engine._ocr() is None
    assert not engine.ocr_available
    assert len(said) == 1, f"warned {len(said)} times for one fault"
    assert "nothing installed" in said[0]


def test_the_engine_passes_the_requested_backend_through(monkeypatch):
    from app.anpr.engine import ANPREngine

    asked = []

    def record(name=None):
        asked.append(name)
        return Loads(), None

    monkeypatch.setattr(readers, "build", record)
    ANPREngine(ocr_backend="easyocr")._ocr()
    assert asked == ["easyocr"]


def test_the_pipeline_hands_its_configured_backend_to_the_engine(tmp_path):
    from app.config import SurveillanceConfig
    from app.pipeline import SurveillancePipeline

    config = SurveillanceConfig(
        source="synthetic", detector="sim", synthetic_frames=3, ocr_backend="easyocr",
        zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, color=False, quiet=True, print_summary=False,
    )
    assert SurveillancePipeline(config).anpr.ocr_backend == "easyocr"


# ------------------------------- reading a plate vs reading a whole vehicle


def test_a_located_plate_and_a_whole_vehicle_use_different_recognisers():
    """They fail at opposite things, so one reader cannot serve both.

    The default has no text-detection stage: on a plate crop it is the most
    accurate backend measured, and handed a picture of a bus it cannot find the
    plate at all. Feeding it the whole vehicle regardless - which this did at
    first - silently stopped plate reading on any footage where the plate model
    missed. The demo clip was one, and nothing failed loudly to say so.
    """
    assert readers.PREFERENCE[0] == "paddle-rec", "the plate-crop reader"
    assert readers.SCENE_PREFERENCE[0] == "easyocr", "the find-it-yourself reader"
    assert readers.PREFERENCE[0] not in readers.SCENE_PREFERENCE, (
        "a recogniser with no detection stage must never be the scene reader"
    )


def test_the_scene_reader_falls_back_too(monkeypatch):
    fallback = Loads()
    monkeypatch.setitem(readers.BACKENDS, "easyocr", Fails)
    monkeypatch.setitem(readers.BACKENDS, "paddle", lambda: fallback)

    reader, problem = readers.build_scene()
    assert reader is fallback and problem is None


def test_with_no_scene_reader_the_reason_is_reported(monkeypatch):
    for candidate in readers.SCENE_PREFERENCE:
        monkeypatch.setitem(readers.BACKENDS, candidate, Fails)

    reader, problem = readers.build_scene()
    assert reader is None
    for candidate in readers.SCENE_PREFERENCE:
        assert candidate in problem


def test_the_engine_asks_for_the_scene_reader_only_when_told_to(monkeypatch):
    from app.anpr.engine import ANPREngine

    plate_reader, scene_reader = Loads([("box", "PLATE", 0.9)]), Loads([("box", "SCENE", 0.9)])
    monkeypatch.setattr(readers, "build", lambda name=None: (plate_reader, None))
    monkeypatch.setattr(readers, "build_scene", lambda: (scene_reader, None))

    engine = ANPREngine()
    image = np.zeros((40, 140, 3), np.uint8)

    assert engine._detect_text(image, localised=True)[0][1] == "PLATE"
    assert engine._detect_text(image, localised=False)[0][1] == "SCENE"


def test_a_missing_scene_reader_warns_once(monkeypatch):
    """Otherwise every unlocated vehicle fails in silence."""
    from app import ui
    from app.anpr.engine import ANPREngine

    said = []
    monkeypatch.setattr(ui, "warn", said.append)
    monkeypatch.setattr(readers, "build_scene", lambda: (None, "none installed"))

    engine = ANPREngine()
    image = np.zeros((40, 140, 3), np.uint8)
    for _ in range(3):
        assert engine._detect_text(image, localised=False) == []

    assert len(said) == 1, f"warned {len(said)} times"
    assert "none installed" in said[0]
