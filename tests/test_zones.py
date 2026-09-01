
import pytest

from app.zones.manager import ZoneManager

CONFIG = {
    "zones": [
        {
            "name": "RESTRICTED",
            "kind": "RESTRICTED",
            "base_risk": 60,
            "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
        },
        {
            "name": "WATCH",
            "kind": "WATCH",
            "base_risk": 30,
            "polygon": [[0.3, 0.0], [0.7, 0.0], [0.7, 1.0], [0.3, 1.0]],
        },
    ]
}


@pytest.fixture
def manager():
    return ZoneManager.from_dict(CONFIG)


def test_locate_outside_all_zones(manager):
    assert manager.locate((0.1, 0.5)) is None


def test_locate_single_zone(manager):
    assert manager.locate((0.35, 0.5)).name == "WATCH"


def test_overlap_resolves_to_highest_priority(manager):
    """0.6 sits in both zones; RESTRICTED must win."""
    assert manager.locate((0.6, 0.5)).name == "RESTRICTED"


def test_unknown_zone_kind_is_rejected():
    bad = {"zones": [{"name": "X", "kind": "NUCLEAR", "polygon": [[0, 0], [1, 0], [1, 1]]}]}
    with pytest.raises(ValueError, match="unknown zone kind"):
        ZoneManager.from_dict(bad)


def test_empty_configuration_is_rejected():
    with pytest.raises(ValueError, match="no zones"):
        ZoneManager.from_dict({"zones": []})


def test_zone_pixels_scale_with_resolution(manager):
    zone = manager.locate((0.6, 0.5))
    assert zone.pixels(1000, 500)[0] == (500, 0)
    assert zone.pixels(500, 250)[0] == (250, 0)


def test_shipped_config_loads():
    manager = ZoneManager.from_file("config/zones.json")
    assert {z.name for z in manager} == {"RESTRICTED", "WATCH"}
