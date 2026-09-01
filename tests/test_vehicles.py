"""Vehicle detection and classification.

YOLO already recognises vehicles; the earlier stages simply discarded them.
These tests cover the three things that had to change to let them through: a
class registry, zone rules that know what they are fencing, and behaviours that
only apply to the object kind they were written for.
"""

import pytest

from app.behaviour.engine import (
    APPROACHING_CAMERA,
    CAMERA_TAMPERING,
    RUNNING,
    BehaviourEngine,
)
from app.config import SurveillanceConfig
from app.context.engine import ContextEngine
from app.context.scene import SceneContext
from app.detection import classes
from app.detection.base import Detection
from app.risk.engine import RiskEngine
from app.zones.manager import Zone, ZoneManager

NIGHT = SceneContext.build("CAM-01", force_night=True)
RESTRICTED = Zone(
    "RESTRICTED", "RESTRICTED", [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0)],
    base_risk=60, description="No-entry strip",
)


# ------------------------------------------------------------------ registry


def test_person_and_vehicles_are_separate_categories():
    assert classes.category_of("PERSON") == classes.PERSON
    for label in ("CAR", "TRUCK", "BUS", "MOTORCYCLE", "BICYCLE"):
        assert classes.category_of(label) == classes.VEHICLE


@pytest.mark.parametrize(
    "selector,expected",
    [
        (["person"], ["PERSON"]),
        (["truck"], ["TRUCK"]),
        (["person", "truck"], ["PERSON", "TRUCK"]),
    ],
)
def test_selectors_resolve_to_labels(selector, expected):
    assert classes.resolve(selector) == expected


def test_the_vehicle_selector_covers_every_vehicle():
    assert set(classes.resolve(["vehicle"])) == set(classes.CATEGORIES[classes.VEHICLE])
    assert "PERSON" not in classes.resolve(["vehicle"])


def test_all_covers_everything():
    assert set(classes.resolve(["all"])) == {e.label for e in classes.REGISTRY}


def test_duplicate_selectors_do_not_duplicate_labels():
    assert classes.resolve(["person", "person", "all"]).count("PERSON") == 1


def test_an_unknown_selector_is_rejected_with_help():
    with pytest.raises(ValueError, match="unknown object selector"):
        classes.resolve(["aeroplane"])


def test_selectors_map_to_coco_ids():
    assert classes.coco_ids(["PERSON"]) == [0]
    assert set(classes.coco_ids(classes.resolve(["vehicle"]))) == {1, 2, 3, 5, 7}


def test_config_resolves_labels_up_front(tmp_path):
    config = SurveillanceConfig(detect=("vehicle",), events_dir=tmp_path, evidence_dir=tmp_path)
    assert "TRUCK" in config.labels
    assert "PERSON" not in config.labels


def test_a_bad_detect_selector_fails_at_configuration_time(tmp_path):
    with pytest.raises(ValueError, match="unknown object selector"):
        SurveillanceConfig(detect=("submarine",), events_dir=tmp_path, evidence_dir=tmp_path)


# ---------------------------------------------------------------------- risk


def test_a_heavier_vehicle_scores_higher_than_a_person():
    engine = RiskEngine()
    scores = {}
    for label in ("BICYCLE", "PERSON", "CAR", "TRUCK"):
        detection = Detection(label, 0.90, (10, 10, 60, 200), track_id=1)
        scores[label] = engine.assess(detection, RESTRICTED, NIGHT).score

    assert scores["BICYCLE"] < scores["PERSON"] < scores["CAR"] < scores["TRUCK"]


def test_the_weighting_is_named_in_the_factor():
    detection = Detection("TRUCK", 0.90, (10, 10, 60, 200), track_id=1)
    assessment = RiskEngine().assess(detection, RESTRICTED, NIGHT)
    zone_factor = assessment.factors[0]
    assert "truck" in zone_factor.detail
    assert "x1.2" in zone_factor.detail


def test_a_person_carries_no_weighting_note():
    detection = Detection("PERSON", 0.90, (10, 10, 60, 200), track_id=1)
    assessment = RiskEngine().assess(detection, RESTRICTED, NIGHT)
    assert "weight" not in assessment.factors[0].detail


def test_the_score_still_equals_its_factors():
    detection = Detection("TRUCK", 0.90, (10, 10, 60, 200), track_id=1)
    assessment = RiskEngine().assess(detection, RESTRICTED, NIGHT)
    assert assessment.score == min(100, sum(f.points for f in assessment.factors))


# --------------------------------------------------------------------- zones


def test_a_zone_covers_both_kinds_by_default():
    zone = ZoneManager.from_dict({
        "zones": [{"name": "R", "kind": "RESTRICTED",
                   "polygon": [[0.5, 0], [1, 0], [1, 1]]}]
    }).zones[0]
    assert zone.applies_to("PERSON")
    assert zone.applies_to("TRUCK")


def test_a_vehicle_gate_fences_people_but_not_vehicles():
    """The lane exists to admit trucks; only a person there is an intrusion."""
    zone = ZoneManager.from_dict({
        "zones": [{"name": "GATE", "kind": "RESTRICTED", "objects": ["PERSON"],
                   "polygon": [[0.5, 0], [1, 0], [1, 1]]}]
    }).zones[0]
    assert zone.applies_to("PERSON")
    assert not zone.applies_to("TRUCK")


def test_a_vehicle_only_zone_ignores_pedestrians():
    zone = ZoneManager.from_dict({
        "zones": [{"name": "LANE", "kind": "RESTRICTED", "objects": ["vehicle"],
                   "polygon": [[0.5, 0], [1, 0], [1, 1]]}]
    }).zones[0]
    assert zone.applies_to("CAR")
    assert not zone.applies_to("PERSON")


def test_the_pipeline_ignores_a_zone_that_does_not_cover_the_object(tmp_path):
    from app.pipeline import SurveillancePipeline

    zones = tmp_path / "gate.json"
    zones.write_text(
        '{"zones":[{"name":"GATE","kind":"RESTRICTED","objects":["VEHICLE"],'
        '"polygon":[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]}]}',
        encoding="utf-8",
    )
    config = SurveillanceConfig(
        source="synthetic", detector="sim", zones_path=zones,
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, color=False,
    )
    pipeline = SurveillancePipeline(config)

    person = Detection("PERSON", 0.9, (400, 100, 500, 400), track_id=1)
    truck = Detection("TRUCK", 0.9, (400, 100, 500, 400), track_id=2)
    assert pipeline._locate(person, 960, 540) is None
    assert pipeline._locate(truck, 960, 540) is not None


# ---------------------------------------------------------------- behaviour


def track_for(label, speed_mps=None, height_ratio=0.0):
    zones = ZoneManager.from_dict({
        "zones": [{"name": "R", "kind": "RESTRICTED",
                   "polygon": [[0.5, 0], [1, 0], [1, 1]]}]
    })
    engine = ContextEngine(zones)
    bbox = (0.4, max(0.0, 0.95 - height_ratio), 0.6, 0.95)
    context = None
    for i in range(6):
        context = engine.observe(1, (0.2, 0.9), None, NIGHT, i * 0.5, bbox_norm=bbox)
    if speed_mps is not None:
        from dataclasses import replace

        context = replace(context, speed_mps=speed_mps)
    return context


def test_a_vehicle_filling_the_frame_is_not_camera_tampering():
    """A truck close to the lens is a truck driving past, not interference."""
    engine = BehaviourEngine()
    close = track_for("TRUCK", height_ratio=0.85)
    assert CAMERA_TAMPERING not in {b.name for b in engine.classify(close, "TRUCK")}
    assert CAMERA_TAMPERING in {b.name for b in engine.classify(close, "PERSON")}


def test_a_vehicle_growing_in_frame_is_not_approaching_the_camera():
    engine = BehaviourEngine()
    zones = ZoneManager.from_dict({
        "zones": [{"name": "R", "kind": "RESTRICTED",
                   "polygon": [[0.5, 0], [1, 0], [1, 1]]}]
    })
    context_engine = ContextEngine(zones)
    found = []
    for i, ratio in enumerate([0.30, 0.36, 0.42, 0.48]):
        bbox = (0.4, 0.95 - ratio, 0.6, 0.95)
        ctx = context_engine.observe(1, (0.2, 0.9), None, NIGHT, i * 0.5, bbox_norm=bbox)
        found = engine.classify(ctx, "CAR")
    assert APPROACHING_CAMERA not in {b.name for b in found}


def test_speed_thresholds_differ_by_object_kind():
    """2 m/s is running for a person and crawling for a car."""
    engine = BehaviourEngine(running_speed_mps=2.0, vehicle_speed_mps=11.0)
    moderate = track_for("CAR", speed_mps=4.0)

    assert RUNNING in {b.name for b in engine.classify(moderate, "PERSON")}
    assert RUNNING not in {b.name for b in engine.classify(moderate, "CAR")}


def test_a_genuinely_fast_vehicle_is_flagged():
    engine = BehaviourEngine(vehicle_speed_mps=11.0)
    speeding = track_for("CAR", speed_mps=15.0)
    found = engine.classify(speeding, "CAR")
    assert RUNNING in {b.name for b in found}
    assert "car" in next(b for b in found if b.name == RUNNING).detail
