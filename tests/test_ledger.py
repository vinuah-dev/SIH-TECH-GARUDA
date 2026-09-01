"""Cross-camera plate resolution.

A border post has many cameras and a vehicle passes several of them. Any one
camera gets a bad angle, a blurred frame, or a plate hidden behind a tow bar -
but rarely do all of them. These tests cover pooling those reads, and handing a
number back to the camera that missed it.

The handoff is inference, not observation, so the tests also pin down that it
stays labelled as such and that it refuses when the evidence is thin.
"""

import numpy as np
import pytest

from app.anpr.ledger import PlateLedger, Sighting
from app.detection.appearance import body_box, describe


def look(colour, size=(400, 600)):
    """An appearance descriptor for a distinctly coloured vehicle."""
    height, width = size
    frame = np.full((height, width, 3), 40, np.uint8)
    frame[100:340, 150:480] = colour
    return describe(frame, (150, 100, 480, 340), region=body_box)


def two_tone(first, second):
    """A vehicle whose descriptor only partly overlaps a plain one."""
    frame = np.full((400, 600, 3), 40, np.uint8)
    frame[100:340, 150:315] = first
    frame[100:340, 315:480] = second
    return describe(frame, (150, 100, 480, 340), region=body_box)


RED = look((60, 60, 200))
RED_AGAIN = look((66, 64, 208))     # same vehicle, other camera, other light
BLUE = look((200, 80, 50))
PART_RED = two_tone((60, 60, 200), (200, 80, 50))   # only half matches RED


@pytest.fixture
def ledger():
    return PlateLedger(handoff_window=180.0)


# ------------------------------------------------------------------ pooling


def test_a_read_is_shared_with_the_whole_fleet(ledger):
    resolved = ledger.record_read("CAM-01", 1, "MH12AB1234", 0.8, at=10.0)
    assert resolved.text == "MH12AB1234"
    assert resolved.read_by == "CAM-01"
    assert not resolved.inferred
    assert ledger.resolve("CAM-01", 1).text == "MH12AB1234"


def test_votes_accumulate_across_cameras(ledger):
    """Two weak reads on different cameras are together a confident one."""
    ledger.record_read("CAM-01", 1, "MH12AB1234", 0.45, at=10.0)
    resolved = ledger.record_read("CAM-03", 5, "MH12AB1234", 0.62, at=25.0)

    assert resolved.reads == 2
    assert set(resolved.cameras) == {"CAM-01", "CAM-03"}
    assert resolved.confidence == pytest.approx(0.62)


def test_each_camera_track_resolves_separately(ledger):
    ledger.record_read("CAM-01", 1, "MH12AB1234", 0.8, at=10.0)
    ledger.record_read("CAM-02", 4, "DL8CAF5030", 0.8, at=12.0)

    assert ledger.resolve("CAM-01", 1).text == "MH12AB1234"
    assert ledger.resolve("CAM-02", 4).text == "DL8CAF5030"
    assert ledger.resolve("CAM-09", 9) is None
    assert ledger.resolve("CAM-01", None) is None


def test_known_plates_lists_what_the_fleet_has_seen(ledger):
    ledger.record_read("CAM-01", 1, "MH12AB1234", 0.8, at=10.0)
    ledger.record_read("CAM-02", 2, "MH12AB1234", 0.8, at=12.0)
    ledger.record_read("CAM-02", 3, "DL8CAF5030", 0.8, at=13.0)

    assert ledger.known_plates() == {"MH12AB1234": 2, "DL8CAF5030": 1}
    assert set(ledger.cameras_for("MH12AB1234")) == {"CAM-01", "CAM-02"}
    assert ledger.cameras_for("XX00XX0000") == ()


# ----------------------------------------------------------------- handoff


def test_a_camera_that_could_not_read_gets_the_number_from_one_that_could(ledger):
    """The whole point: CAM-01 missed it, CAM-03 got it."""
    ledger.note_sighting(Sighting("CAM-01", 1, at=10.0, appearance=RED))
    assert ledger.resolve("CAM-01", 1) is None

    ledger.record_read("CAM-03", 7, "MH12AB1234", 0.9, at=40.0, appearance=RED_AGAIN)

    handed = ledger.resolve("CAM-01", 1)
    assert handed.text == "MH12AB1234"
    assert handed.read_by == "CAM-03"
    assert handed.inferred, "a handoff is inference and must say so"
    assert ledger.handoffs == 1


def test_a_different_vehicle_is_not_given_the_plate(ledger):
    ledger.note_sighting(Sighting("CAM-01", 1, at=10.0, appearance=BLUE))
    ledger.record_read("CAM-03", 7, "MH12AB1234", 0.9, at=40.0, appearance=RED)

    assert ledger.resolve("CAM-01", 1) is None
    assert ledger.handoffs == 0


def test_a_sighting_too_long_ago_is_not_linked(ledger):
    """Beyond a plausible travel time it is a different vehicle."""
    ledger.note_sighting(Sighting("CAM-01", 1, at=10.0, appearance=RED))
    ledger.record_read("CAM-03", 7, "MH12AB1234", 0.9, at=10_000.0, appearance=RED_AGAIN)
    assert ledger.resolve("CAM-01", 1) is None


def test_a_camera_never_hands_a_plate_to_itself(ledger):
    """It would have used its own read; this would only mask a tracking split."""
    ledger.note_sighting(Sighting("CAM-01", 1, at=10.0, appearance=RED))
    ledger.record_read("CAM-01", 2, "MH12AB1234", 0.9, at=20.0, appearance=RED_AGAIN)
    assert ledger.resolve("CAM-01", 1) is None


def test_no_appearance_means_no_handoff(ledger):
    ledger.note_sighting(Sighting("CAM-01", 1, at=10.0, appearance=None))
    ledger.record_read("CAM-03", 7, "MH12AB1234", 0.9, at=20.0, appearance=RED)
    assert ledger.resolve("CAM-01", 1) is None


def test_a_marginal_match_is_accepted_or_refused_by_the_threshold():
    """A half-matching vehicle is exactly the case the threshold governs."""
    from app.detection.appearance import similarity

    overlap = similarity(RED, PART_RED)
    assert 0.2 < overlap < 0.8, "test rig should produce a genuinely marginal pair"

    lenient = PlateLedger(handoff_similarity=overlap - 0.05)
    lenient.note_sighting(Sighting("CAM-01", 1, at=10.0, appearance=RED))
    lenient.record_read("CAM-03", 7, "MH12AB1234", 0.9, at=20.0, appearance=PART_RED)
    assert lenient.resolve("CAM-01", 1) is not None

    strict = PlateLedger(handoff_similarity=overlap + 0.05)
    strict.note_sighting(Sighting("CAM-01", 1, at=10.0, appearance=RED))
    strict.record_read("CAM-03", 7, "MH12AB1234", 0.9, at=20.0, appearance=PART_RED)
    assert strict.resolve("CAM-01", 1) is None


def test_an_own_read_is_never_replaced_by_a_handoff(ledger):
    ledger.record_read("CAM-01", 1, "MH12AB1234", 0.9, at=10.0, appearance=RED)
    ledger.record_read("CAM-03", 7, "DL8CAF5030", 0.9, at=20.0, appearance=RED_AGAIN)

    resolved = ledger.resolve("CAM-01", 1)
    assert resolved.text == "MH12AB1234"
    assert not resolved.inferred


# ------------------------------------------------------------------ upkeep


def test_a_track_with_a_plate_stops_being_a_pending_sighting(ledger):
    ledger.note_sighting(Sighting("CAM-01", 1, at=10.0, appearance=RED))
    assert ledger.pending_sightings == 1

    ledger.record_read("CAM-01", 1, "MH12AB1234", 0.9, at=12.0)
    assert ledger.pending_sightings == 0


def test_repeated_sightings_of_one_track_are_not_duplicated(ledger):
    for at in (10.0, 11.0, 12.0):
        ledger.note_sighting(Sighting("CAM-01", 1, at=at, appearance=RED))
    assert ledger.pending_sightings == 1


def test_stale_sightings_are_forgotten(ledger):
    ledger.note_sighting(Sighting("CAM-01", 1, at=10.0, appearance=RED))
    ledger.note_sighting(Sighting("CAM-02", 2, at=10_000.0, appearance=BLUE))
    assert ledger.pending_sightings == 1


def test_reset_clears_everything(ledger):
    ledger.note_sighting(Sighting("CAM-01", 1, at=10.0, appearance=RED))
    ledger.record_read("CAM-03", 7, "MH12AB1234", 0.9, at=20.0, appearance=RED_AGAIN)
    ledger.reset()

    assert ledger.known_plates() == {}
    assert ledger.pending_sightings == 0
    assert ledger.handoffs == 0


def test_concurrent_cameras_do_not_corrupt_the_ledger():
    """One ledger, one thread per camera - the fleet's actual shape."""
    import threading

    ledger = PlateLedger()
    errors: list[Exception] = []

    def camera(name: str) -> None:
        try:
            for i in range(40):
                ledger.record_read(name, i, f"MH12AB{1000 + i}", 0.8, at=float(i))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=camera, args=(f"CAM-0{n}",)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(ledger.known_plates()) == 40
    assert sum(ledger.known_plates().values()) == 160


# ------------------------------------------------------------- integration


def test_the_fleet_runner_gives_every_camera_the_same_ledger(tmp_path):
    from app.config import SurveillanceConfig
    from app.runner import CameraSpec, MultiCameraRunner

    config = SurveillanceConfig(
        source="synthetic", detector="sim", synthetic_frames=20,
        zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, color=False,
    )
    specs = [
        CameraSpec(camera_id="CAM-01", source="synthetic"),
        CameraSpec(camera_id="CAM-02", source="synthetic"),
    ]
    runner = MultiCameraRunner(specs, config)
    runner.detector = None
    workers = runner._build_workers()

    ledgers = {id(w.pipeline.anpr.ledger) for w in workers}
    assert len(ledgers) == 1
    assert workers[0].pipeline.anpr.ledger is runner.plate_ledger
    # Each camera still identifies itself when writing into it.
    assert {w.pipeline.anpr.camera_id for w in workers} == {"CAM-01", "CAM-02"}
