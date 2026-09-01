"""Vehicle registry: does the plate match the vehicle carrying it?

A plate registered to a motorcycle, bolted to a truck, is the signature of a
cloned or transplanted plate. These tests cover the comparison, and - just as
importantly - the cases where it must stay quiet, because a registry extract is
always incomplete and crying wolf on every unlisted vehicle is useless.
"""

import json

import pytest

from app.behaviour.engine import (
    PLATE_MISMATCH,
    UNREGISTERED,
    WATCHLISTED,
    BehaviourEngine,
)
from app.registry import VehicleRegistry, canonical_class


def registry_file(tmp_path, vehicles):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"vehicles": vehicles}), encoding="utf-8")
    return path


@pytest.fixture
def registry(tmp_path):
    return VehicleRegistry.from_file(registry_file(tmp_path, [
        {"plate": "MH12AB1234", "vehicle_class": "Motor Car", "make": "Maruti",
         "colour": "White", "status": "ACTIVE"},
        {"plate": "DL8CAF5030", "vehicle_class": "Motorcycle", "status": "ACTIVE"},
        {"plate": "UP16BX9999", "vehicle_class": "Goods Carrier", "status": "STOLEN"},
        {"plate": "PB10CD4567", "vehicle_class": "Spacecraft", "status": "ACTIVE"},
    ]))


# --------------------------------------------------------- class vocabulary


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Motor Car", "CAR"), ("LMV", "CAR"), ("SUV", "CAR"),
        ("Motorcycle", "MOTORCYCLE"), ("Scooter", "MOTORCYCLE"), ("2W", "MOTORCYCLE"),
        ("Goods Carrier", "TRUCK"), ("HGV", "TRUCK"), ("Tanker", "TRUCK"),
        ("Omnibus", "BUS"), ("Minibus", "BUS"),
    ],
)
def test_rto_wording_maps_onto_detector_classes(raw, expected):
    """RTO paperwork and COCO do not use the same words for the same thing."""
    assert canonical_class(raw) == expected


def test_unknown_wording_maps_to_nothing():
    assert canonical_class("Spacecraft") is None
    assert canonical_class(None) is None


# ------------------------------------------------------------------ loading


def test_a_registry_loads_from_file(registry):
    assert len(registry.backend) == 4
    assert registry.enabled


def test_lookup_ignores_spacing_and_case(registry):
    assert registry.backend.lookup("mh 12 ab 1234").plate == "MH12AB1234"


def test_a_missing_registry_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        VehicleRegistry.from_file(tmp_path / "nope.json")


def test_no_registry_means_no_checks():
    assert not VehicleRegistry().enabled
    assert VehicleRegistry().check("MH12AB1234", "CAR") is None


def test_the_shipped_sample_registry_loads():
    assert len(VehicleRegistry.from_file("config/vehicle-registry.json").backend) >= 3


# ------------------------------------------------------------------ checks


def test_a_matching_vehicle_raises_nothing(registry):
    check = registry.check("MH12AB1234", "CAR")
    assert check.known and not check.mismatch and not check.flagged
    assert BehaviourEngine().from_registry(check) == []


def test_a_plate_on_the_wrong_vehicle_is_a_mismatch(registry):
    """The whole point: a motorcycle plate bolted to a bus."""
    check = registry.check("DL8CAF5030", "BUS")
    assert check.mismatch
    assert "registered to a motorcycle" in check.detail
    assert "sees a bus" in check.detail

    found = BehaviourEngine().from_registry(check)
    assert [b.name for b in found] == [PLATE_MISMATCH]
    assert found[0].points > 0


def test_a_watchlisted_vehicle_outranks_a_mismatch(registry):
    check = registry.check("UP16BX9999", "CAR")
    assert check.flagged
    found = BehaviourEngine().from_registry(check)
    assert [b.name for b in found] == [WATCHLISTED]
    assert found[0].points > BehaviourEngine().plate_mismatch_points


def test_an_unknown_plate_is_noted_but_only_just(registry):
    """An extract is always incomplete; absence is weak evidence."""
    check = registry.check("XX99XX9999", "CAR")
    assert not check.known and not check.mismatch

    engine = BehaviourEngine()
    found = engine.from_registry(check)
    assert [b.name for b in found] == [UNREGISTERED]
    assert found[0].points < engine.plate_mismatch_points


def test_wording_the_registry_uses_but_we_do_not_know_is_not_a_mismatch(registry):
    """Otherwise every unusual vehicle type would cry wolf."""
    check = registry.check("PB10CD4567", "TRUCK")
    assert check.known
    assert not check.mismatch
    assert BehaviourEngine().from_registry(check) == []


def test_people_are_never_checked_against_the_registry(registry):
    assert registry.check("MH12AB1234", "PERSON") is None


def test_no_plate_means_no_check(registry):
    assert registry.check("", "CAR") is None


def test_a_check_serialises_without_owner_details(registry):
    """Registry extracts carry personal data; events should not."""
    payload = registry.check("MH12AB1234", "CAR").to_dict()
    assert set(payload) == {
        "plate", "observed_class", "known", "mismatch", "flagged", "detail", "record",
    }
    assert set(payload["record"]) == {
        "plate", "vehicle_class", "make", "model", "colour", "status", "flagged", "note",
    }
    assert "owner" not in json.dumps(payload).lower()


# ------------------------------------------------------------- integration


def test_the_pipeline_flags_a_cloned_plate(tmp_path, monkeypatch):
    from app.anpr.engine import ANPREngine, PlateReading
    from app.config import SurveillanceConfig
    from app.detection.base import Detection
    from app.pipeline import SurveillancePipeline

    path = registry_file(tmp_path, [
        {"plate": "DL8CAF5030", "vehicle_class": "Motorcycle", "status": "ACTIVE"},
    ])
    config = SurveillanceConfig(
        source="synthetic", detector="sim", zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        registry_path=path, save_clips=False, color=False,
    )
    pipeline = SurveillancePipeline(config)
    monkeypatch.setattr(
        ANPREngine, "best_for",
        lambda self, track_id: PlateReading("DL8CAF5030", 0.9, (0, 0, 1, 1), reads=3),
    )

    bus = Detection("BUS", 0.9, (10, 10, 100, 200), track_id=1)
    check = pipeline._verify_plate(bus)
    assert check.mismatch

    behaviours = pipeline.behaviour.from_registry(check)
    assert [b.name for b in behaviours] == [PLATE_MISMATCH]


def test_a_plate_is_only_checked_once_per_track(tmp_path, monkeypatch):
    """The plate does not change while we watch the vehicle."""
    from app.anpr.engine import ANPREngine, PlateReading
    from app.config import SurveillanceConfig
    from app.detection.base import Detection
    from app.pipeline import SurveillancePipeline

    path = registry_file(tmp_path, [
        {"plate": "MH12AB1234", "vehicle_class": "Motor Car", "status": "ACTIVE"},
    ])
    config = SurveillanceConfig(
        source="synthetic", detector="sim", zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        registry_path=path, save_clips=False, color=False,
    )
    pipeline = SurveillancePipeline(config)
    monkeypatch.setattr(
        ANPREngine, "best_for",
        lambda self, track_id: PlateReading("MH12AB1234", 0.9, (0, 0, 1, 1), reads=3),
    )

    calls = {"n": 0}
    original = pipeline.registry.check

    def counted(plate, observed):
        calls["n"] += 1
        return original(plate, observed)

    pipeline.registry.check = counted
    car = Detection("CAR", 0.9, (10, 10, 100, 200), track_id=1)
    for _ in range(5):
        pipeline._verify_plate(car)
    assert calls["n"] == 1
