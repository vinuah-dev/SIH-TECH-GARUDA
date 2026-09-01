"""Following one person across the cameras of a post.

A vehicle carries its identity on the outside; a person carries nothing. All
that is left is appearance, which is weak - the same coat is a different colour
under sodium light than under infrared, and two guards in one uniform look
alike at any distance.

So these tests are mostly about the ledger *refusing* to link: overlapping
sightings, implausible travel times, one camera seeing two people. Getting a
link wrong writes a movement history that never happened.
"""

import threading

import numpy as np
import pytest

from app.detection.appearance import body_box, describe, similarity
from app.tracking import PersonLedger, Subject


def look(colour, patch=None):
    """An appearance descriptor for a distinctly dressed person."""
    frame = np.full((400, 600, 3), 40, np.uint8)
    frame[100:340, 150:480] = colour
    if patch:
        frame[100:340, 150:315] = patch
    return describe(frame, (150, 100, 480, 340), region=body_box)


RED = look((60, 60, 200))
RED_AGAIN = look((66, 64, 208))          # same person, other camera
BLUE = look((200, 80, 50))
HALF_RED = look((60, 60, 200), patch=(200, 80, 50))    # only partly alike


@pytest.fixture
def ledger():
    return PersonLedger(travel_window=300.0, min_travel_seconds=1.0)


# ------------------------------------------------------------------ linking


def test_one_person_on_one_camera_is_one_subject(ledger):
    first = ledger.observe("CAM-01", 1, 10.0, RED)
    again = ledger.observe("CAM-01", 1, 12.0, RED)

    assert first.subject_id == again.subject_id
    assert ledger.subjects == 1
    assert again.cameras == ("CAM-01",)


def test_the_same_person_on_a_second_camera_is_linked(ledger):
    """The whole point: has this person been seen elsewhere on this post?"""
    ledger.observe("CAM-01", 1, 10.0, RED)
    later = ledger.observe("CAM-03", 7, 60.0, RED_AGAIN)

    assert set(later.cameras) == {"CAM-01", "CAM-03"}
    assert ledger.subjects == 1
    assert ledger.links == 1


def test_a_link_is_labelled_as_inferred(ledger):
    """An operator must be able to tell an observation from a deduction."""
    ledger.observe("CAM-01", 1, 10.0, RED)
    subject = ledger.observe("CAM-03", 7, 60.0, RED_AGAIN)

    payload = subject.to_dict()
    assert payload["inferred"] is True
    assert payload["sightings"] == 2
    assert set(payload["cameras"]) == {"CAM-01", "CAM-03"}


def test_a_single_camera_subject_is_not_inferred(ledger):
    subject = ledger.observe("CAM-01", 1, 10.0, RED)
    assert subject.to_dict()["inferred"] is False


# ---------------------------------------------------------------- refusing


def test_somebody_dressed_differently_is_a_different_subject(ledger):
    ledger.observe("CAM-01", 1, 10.0, RED)
    other = ledger.observe("CAM-03", 7, 60.0, BLUE)

    assert other.cameras == ("CAM-03",)
    assert ledger.subjects == 2
    assert ledger.links == 0


def test_overlapping_sightings_are_two_people(ledger):
    """If CAM-01 can still see them while CAM-03 starts to, there are two."""
    ledger.observe("CAM-01", 1, 10.0, RED)
    simultaneous = ledger.observe("CAM-03", 7, 10.2, RED_AGAIN)

    assert simultaneous.cameras == ("CAM-03",)
    assert ledger.links == 0


def test_overlapping_cameras_can_be_declared():
    """A post whose cameras genuinely overlap says so, and linking resumes."""
    overlapping = PersonLedger(min_travel_seconds=0.0)
    overlapping.observe("CAM-01", 1, 10.0, RED)
    subject = overlapping.observe("CAM-03", 7, 10.2, RED_AGAIN)

    assert set(subject.cameras) == {"CAM-01", "CAM-03"}


def test_too_long_a_gap_is_a_different_person(ledger):
    """Beyond a plausible walk it is somebody who owns a similar coat."""
    ledger.observe("CAM-01", 1, 10.0, RED)
    much_later = ledger.observe("CAM-03", 7, 10_000.0, RED_AGAIN)

    assert much_later.cameras == ("CAM-03",)
    assert ledger.links == 0


def test_two_tracks_on_one_camera_are_never_merged(ledger):
    """One camera producing two tracks means two people, not one seen twice."""
    ledger.observe("CAM-01", 1, 10.0, RED)
    second = ledger.observe("CAM-01", 2, 30.0, RED_AGAIN)

    assert second.cameras == ("CAM-01",)
    assert ledger.subjects == 2


def test_no_appearance_means_no_link(ledger):
    ledger.observe("CAM-01", 1, 10.0, None)
    alone = ledger.observe("CAM-03", 7, 60.0, RED)
    assert alone.cameras == ("CAM-03",)


def test_an_untracked_detection_is_ignored(ledger):
    assert ledger.observe("CAM-01", None, 10.0, RED) is None


def test_the_threshold_governs_a_marginal_match():
    """A half-matching person is exactly what the threshold is there to judge."""
    overlap = similarity(RED, HALF_RED)
    assert 0.2 < overlap < 0.9, "test rig should produce a genuinely marginal pair"

    lenient = PersonLedger(similarity_threshold=overlap - 0.05)
    lenient.observe("CAM-01", 1, 10.0, RED)
    assert len(lenient.observe("CAM-03", 7, 60.0, HALF_RED).cameras) == 2

    strict = PersonLedger(similarity_threshold=overlap + 0.05)
    strict.observe("CAM-01", 1, 10.0, RED)
    assert strict.observe("CAM-03", 7, 60.0, HALF_RED).cameras == ("CAM-03",)


def test_cross_camera_is_stricter_than_within_camera_reid():
    """Recovering a two-second occlusion is a different problem from this."""
    from app.detection.tracker import IoUTracker

    assert PersonLedger().similarity_threshold > IoUTracker().reid_threshold


# ------------------------------------------------------------------ queries


def test_a_subject_can_be_found_from_either_camera(ledger):
    ledger.observe("CAM-01", 1, 10.0, RED)
    ledger.observe("CAM-03", 7, 60.0, RED_AGAIN)

    from_first = ledger.subject_for("CAM-01", 1)
    from_second = ledger.subject_for("CAM-03", 7)
    assert from_first is not None
    assert from_first.subject_id == from_second.subject_id
    assert ledger.subject_for("CAM-09", 9) is None
    assert ledger.subject_for("CAM-01", None) is None


def test_subjects_seen_on_several_cameras_can_be_listed(ledger):
    ledger.observe("CAM-01", 1, 10.0, RED)
    ledger.observe("CAM-03", 7, 60.0, RED_AGAIN)
    ledger.observe("CAM-01", 2, 20.0, BLUE)

    travelled = ledger.seen_on_multiple_cameras()
    assert len(travelled) == 1
    assert set(travelled[0].cameras) == {"CAM-01", "CAM-03"}


def test_a_subject_records_when_it_was_first_and_last_seen(ledger):
    ledger.observe("CAM-01", 1, 10.0, RED)
    subject = ledger.observe("CAM-03", 7, 60.0, RED_AGAIN)

    assert subject.first_seen == 10.0
    assert subject.last_seen == 60.0


# ------------------------------------------------------------------ upkeep


def test_stale_subjects_are_forgotten():
    ledger = PersonLedger(forget_after=100.0)
    ledger.observe("CAM-01", 1, 10.0, RED)
    ledger.observe("CAM-02", 2, 10_000.0, BLUE)

    assert ledger.subjects == 1
    assert ledger.subject_for("CAM-01", 1) is None


def test_reset_clears_everything(ledger):
    ledger.observe("CAM-01", 1, 10.0, RED)
    ledger.observe("CAM-03", 7, 60.0, RED_AGAIN)
    ledger.reset()

    assert ledger.subjects == 0
    assert ledger.links == 0
    assert ledger.subject_for("CAM-01", 1) is None


def test_concurrent_cameras_do_not_corrupt_the_ledger():
    """One ledger, one thread per camera - the fleet's actual shape."""
    ledger = PersonLedger()
    errors: list[Exception] = []

    def camera(name: str) -> None:
        try:
            for i in range(40):
                ledger.observe(name, i, float(i), None)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=camera, args=(f"CAM-0{n}",)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert ledger.subjects == 160


# ------------------------------------------------------------- integration


def test_the_fleet_gives_every_camera_the_same_person_ledger(tmp_path):
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

    ledgers = {id(w.pipeline.people) for w in workers}
    assert len(ledgers) == 1
    assert workers[0].pipeline.people is runner.person_ledger


def test_a_lone_pipeline_runs_without_a_ledger(tmp_path):
    """One camera has nowhere to follow anyone to, and must not require one."""
    from app.config import SurveillanceConfig
    from app.pipeline import SurveillancePipeline

    config = SurveillanceConfig(
        source="synthetic", detector="sim", synthetic_frames=40, force_night=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, color=False,
    )
    pipeline = SurveillancePipeline(config)
    assert pipeline.people is None
    assert pipeline.run().alerts == 1
