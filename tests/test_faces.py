"""Recognising the people a post expects to see.

The purpose is subtraction: a guard walking his own patrol line trips the same
fence as an intruder, and an operator shown both learns to ignore both. So most
of these tests are about the asymmetry that makes that work - recognition
lowers risk, while *failing* to recognise somebody raises nothing, because
almost nobody at a border is enrolled and never will be.

The model files are downloaded, not committed, so anything needing them skips
rather than fails on a fresh checkout.
"""

import json

import numpy as np
import pytest

from app.behaviour.engine import AUTHORISED, UNRECOGNISED, BehaviourEngine
from app.faces.detector import FaceEngine
from app.faces.engine import FaceRecognitionEngine
from app.faces.roster import DEFAULT_THRESHOLD, FaceRoster, Match, Person

MODELS = FaceEngine()
needs_models = pytest.mark.skipif(
    not MODELS.available, reason="face models not downloaded"
)


def vector(seed: int, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=size).astype(np.float32)
    return v / np.linalg.norm(v)


def cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def person(person_id="SSB-114", seed=1, **kwargs):
    return Person(
        person_id=person_id,
        label=kwargs.pop("label", "Havildar Singh"),
        vectors=(vector(seed),),
        **kwargs,
    )


# ------------------------------------------------------------------- roster


def test_an_enrolled_person_is_recognised():
    roster = FaceRoster(people=[person()])
    match = roster.match(vector(1), cosine)
    assert match.recognised
    assert match.person.person_id == "SSB-114"
    assert match.similarity == pytest.approx(1.0, abs=1e-5)


def test_a_stranger_is_not_recognised():
    roster = FaceRoster(people=[person(seed=1)])
    match = roster.match(vector(99), cosine)
    assert not match.recognised
    assert match.person is None


def test_a_near_miss_below_the_threshold_is_refused():
    """Waving through the wrong person costs more than asking again."""
    roster = FaceRoster(threshold=0.99, people=[person(seed=1)])
    blend = (vector(1) * 0.8 + vector(2) * 0.2)
    assert not roster.match(blend / np.linalg.norm(blend), cosine).recognised


def test_several_angles_all_count_for_one_person():
    """A face vector is sensitive to pose, so people are enrolled more than once."""
    enrolled = Person("SSB-2", "Naik Kumar", (vector(3), vector(4)))
    roster = FaceRoster(people=[enrolled])
    assert roster.match(vector(4), cosine).recognised


def test_an_empty_roster_recognises_nobody():
    assert not FaceRoster().match(vector(1), cosine).recognised


def test_a_missing_vector_recognises_nobody():
    assert not FaceRoster(people=[person()]).match(None, cosine).recognised


# --------------------------------------------------------- roster storage


def test_a_roster_round_trips_through_a_file(tmp_path):
    roster = FaceRoster(people=[person(role="day patrol", note="gate 2")])
    path = roster.save(tmp_path / "roster.json")

    loaded = FaceRoster.from_file(path)
    assert len(loaded) == 1
    assert loaded.people[0].role == "day patrol"
    assert loaded.match(vector(1), cosine).recognised


def test_the_roster_file_stores_no_images(tmp_path):
    """Vectors cannot be turned back into a picture of somebody's face."""
    path = FaceRoster(people=[person()]).save(tmp_path / "roster.json")
    raw = json.loads(path.read_text(encoding="utf-8"))

    entry = raw["people"][0]
    assert set(entry) == {"person_id", "label", "role", "note", "vectors"}

    # Every stored vector is plain numbers - nothing that could be decoded
    # back into a picture. Checked against the person entries rather than the
    # whole file, because the file's own comment says the word "photograph".
    stored = json.dumps(raw["people"]).lower()
    for banned in ("image", "photo", "jpg", "png", "base64"):
        assert banned not in stored
    for vec in entry["vectors"]:
        assert all(isinstance(x, (int, float)) for x in vec)


def test_a_person_without_vectors_is_skipped(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text(
        json.dumps({"people": [{"person_id": "X", "label": "no vectors"}]}),
        encoding="utf-8",
    )
    assert len(FaceRoster.from_file(path)) == 0


def test_a_missing_roster_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        FaceRoster.from_file(tmp_path / "nope.json")


def test_enrolling_the_same_id_twice_replaces_it():
    roster = FaceRoster(people=[person(seed=1)])
    roster.add(person(seed=5, label="corrected"))
    assert len(roster) == 1
    assert roster.people[0].label == "corrected"


# ------------------------------------------------------- what events carry


def test_an_event_carries_the_identity_not_the_biometrics():
    match = Match(person=person(role="day patrol"), similarity=0.61,
                  threshold=DEFAULT_THRESHOLD)
    payload = match.to_dict()

    assert payload["identity"] == {
        "person_id": "SSB-114", "label": "Havildar Singh", "role": "day patrol",
    }
    assert "vectors" not in json.dumps(payload)


def test_an_unrecognised_match_carries_no_identity():
    payload = Match(person=None, similarity=0.2, threshold=0.45).to_dict()
    assert payload["identity"] is None
    assert payload["recognised"] is False


# --------------------------------------------------------------- behaviour


def test_recognising_somebody_lowers_the_risk():
    """A guard on his own patrol line must not read like an intruder."""
    found = BehaviourEngine().from_identity(
        Match(person(role="day patrol"), 0.61, DEFAULT_THRESHOLD)
    )
    assert [b.name for b in found] == [AUTHORISED]
    assert found[0].points < 0
    assert "Havildar Singh" in found[0].detail


def test_being_unrecognised_raises_nothing_by_default():
    """Almost nobody at a border is enrolled; that is not evidence."""
    assert BehaviourEngine().from_identity(Match(None, 0.2, 0.45)) == []


def test_a_checkpoint_can_treat_an_unknown_face_as_a_signal():
    strict = BehaviourEngine(unrecognised_points=8)
    found = strict.from_identity(Match(None, 0.2, 0.45))
    assert [b.name for b in found] == [UNRECOGNISED]
    assert found[0].points > 0


def test_no_face_at_all_says_nothing():
    assert BehaviourEngine().from_identity(None) == []


def test_recognition_can_pull_an_alert_out_of_range():
    """The subtraction has to be large enough to actually matter."""
    from app.context.scene import SceneContext
    from app.detection.base import Detection
    from app.risk.engine import RiskEngine
    from app.zones.manager import Zone

    zone = Zone("RESTRICTED", "RESTRICTED", [(0.5, 0), (1, 0), (1, 1)], base_risk=60)
    night = SceneContext.build("CAM-01", force_night=True)
    detection = Detection("PERSON", 0.9, (10, 10, 60, 200), track_id=1)
    engine = RiskEngine()

    intruder = engine.assess(detection, zone, night)
    guard = engine.assess(
        detection, zone, night, None,
        BehaviourEngine().from_identity(Match(person(), 0.61, DEFAULT_THRESHOLD)),
    )
    assert intruder.severity == "HIGH"
    assert guard.score < intruder.score
    assert guard.severity in ("LOW", "MEDIUM")


# ------------------------------------------------------------------ engine


def test_recognition_is_off_without_a_roster():
    assert not FaceRecognitionEngine(faces=MODELS, roster=None).available
    assert not FaceRecognitionEngine(faces=MODELS, roster=FaceRoster()).available


def test_vehicles_are_never_matched_against_a_face_roster():
    from app.detection.base import Detection

    engine = FaceRecognitionEngine(
        faces=MODELS, roster=FaceRoster(people=[person()])
    )
    truck = Detection("TRUCK", 0.9, (0, 0, 100, 100), track_id=1)
    assert engine.observe(np.zeros((200, 200, 3), np.uint8), truck) is None


def test_one_sighting_is_not_an_identity():
    """A single match is a guess, and acting on a guess waves through the wrong person."""
    from app.faces.engine import _TrackFace

    engine = FaceRecognitionEngine(
        faces=MODELS, roster=FaceRoster(people=[person()]), min_agreeing=2
    )
    state = _TrackFace()
    state.best = Match(person(), 0.7, DEFAULT_THRESHOLD)

    state.votes["SSB-114"] = 1
    assert engine._settled(state) is None, "one sighting must not settle an identity"

    state.votes["SSB-114"] = 2
    settled = engine._settled(state)
    assert settled is not None and settled.person.person_id == "SSB-114"


def test_a_disputed_identity_does_not_settle():
    """Two names for one track means the answer is not known yet."""
    from app.faces.engine import _TrackFace

    engine = FaceRecognitionEngine(
        faces=MODELS, roster=FaceRoster(people=[person()]), min_agreeing=2
    )
    state = _TrackFace()
    state.best = Match(person("SSB-9", label="someone else"), 0.9, DEFAULT_THRESHOLD)
    state.votes["SSB-114"] = 2

    # The most-voted id is not the one the best match names, so nothing settles.
    assert engine._settled(state) is None


# ------------------------------------------------------- needs the models


@needs_models
def test_the_detector_finds_faces_in_the_sample_photo():
    import cv2

    engine = FaceEngine()
    engine.min_width = 15
    faces = engine.detect(cv2.imread("data/samples/bus.jpg"))
    assert faces
    assert all(f.width >= 15 for f in faces)
    # Largest first, so callers can take the nearest face.
    assert faces == sorted(faces, key=lambda f: -f.width)


@needs_models
def test_a_face_embeds_and_matches_itself():
    import cv2

    engine = FaceEngine()
    engine.min_width = 15
    image = cv2.imread("data/samples/bus.jpg")
    face = engine.detect(image)[0]

    vector = engine.embed(image, face)
    assert vector is not None
    assert vector.shape == (128,)
    assert engine.similarity(vector, vector) == pytest.approx(1.0, abs=1e-3)


@needs_models
def test_faces_below_the_minimum_width_are_ignored():
    """Below about 60 px the vector is noise, and matching noise invents people."""
    import cv2

    engine = FaceEngine()
    engine.min_width = 500
    assert engine.detect(cv2.imread("data/samples/bus.jpg")) == []


@needs_models
def test_an_empty_frame_yields_no_faces():
    assert FaceEngine().detect(np.zeros((240, 320, 3), np.uint8)) == []
