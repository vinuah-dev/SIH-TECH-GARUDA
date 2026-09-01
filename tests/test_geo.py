"""Putting the cameras on a map.

An alert that says CAM-05 means nothing to somebody who does not already know
where CAM-05 is. On a map it means *this stretch of the river*, and a commander
can send the nearest patrol instead of asking which camera that was.

Two distinctions run through these tests, because getting either wrong would
put a confident lie on a map:

* a **zone** is in frame coordinates and cannot be drawn on a map at all;
  a **geofence** here is in latitude and longitude and knows nothing of cameras
* a camera's **bearing wedge** is a claim about where it points, never a
  measurement of what it can see - so it is drawn only when surveyed
"""

import json

import pytest

from app.geo import CameraLocation, Geofence, InvalidLocation, SiteMap

GATE = CameraLocation(28.613939, 77.209023, bearing=90, field_of_view=60, site="Gate A")
RIVER = CameraLocation(28.614512, 77.209870, site="River Bank")

SECTOR = Geofence(
    "SECTOR-7",
    ((28.6100, 77.2050), (28.6180, 77.2050), (28.6180, 77.2130), (28.6100, 77.2130)),
)


# ------------------------------------------------------------------ validation


def test_a_coordinate_off_the_earth_is_refused_at_load():
    """Better a startup error than a pin in the Pacific during an incident."""
    with pytest.raises(InvalidLocation):
        CameraLocation(latitude=91.0, longitude=77.0)
    with pytest.raises(InvalidLocation):
        CameraLocation(latitude=28.0, longitude=181.0)


def test_an_impossible_bearing_is_refused():
    with pytest.raises(InvalidLocation):
        CameraLocation(28.0, 77.0, bearing=400)
    with pytest.raises(InvalidLocation):
        CameraLocation(28.0, 77.0, field_of_view=0)


def test_a_camera_with_no_location_stays_none():
    """The normal case today: coordinates arrive when somebody walks the post."""
    assert CameraLocation.from_dict(None) is None
    assert CameraLocation.from_dict({}) is None


def test_a_location_missing_its_coordinates_is_an_error_not_a_silent_skip():
    with pytest.raises(InvalidLocation):
        CameraLocation.from_dict({"site": "Gate A"})


def test_either_spelling_of_the_coordinates_is_accepted():
    short = CameraLocation.from_dict({"lat": 28.5, "lon": 77.1})
    long = CameraLocation.from_dict({"latitude": 28.5, "longitude": 77.1})
    assert short == long


# -------------------------------------------------------------------- geometry


def test_the_distance_between_two_cameras_is_in_metres():
    metres = GATE.distance_to(RIVER)
    assert 90 < metres < 120, f"got {metres} m"
    assert GATE.distance_to(GATE) == pytest.approx(0.0, abs=0.01)


def test_distance_is_the_same_measured_either_way():
    assert GATE.distance_to(RIVER) == pytest.approx(RIVER.distance_to(GATE), abs=0.01)


def test_distance_is_not_flat_trigonometry():
    """A post may be surveyed in the same file as one far away, where flat maths lies."""
    delhi = CameraLocation(28.6139, 77.2090)
    mumbai = CameraLocation(19.0760, 72.8777)
    assert 1_100_000 < delhi.distance_to(mumbai) < 1_200_000


def test_a_surveyed_camera_draws_a_coverage_wedge():
    wedge = GATE.coverage_wedge(reach_m=120)
    assert len(wedge) > 3
    # It starts and ends at the camera, so the polygon closes on the lens.
    assert wedge[0] == wedge[-1] == [GATE.latitude, GATE.longitude]
    # Facing east, every point of the arc is east of the camera.
    assert all(point[1] > GATE.longitude for point in wedge[1:-1])


def test_an_unsurveyed_bearing_draws_no_wedge():
    """A wedge from a guessed bearing points confidently at the wrong hillside."""
    assert RIVER.coverage_wedge() == []
    assert CameraLocation(28.0, 77.0, bearing=90).coverage_wedge() == []
    assert CameraLocation(28.0, 77.0, field_of_view=60).coverage_wedge() == []


def test_the_wedge_reaches_about_as_far_as_asked():
    wedge = GATE.coverage_wedge(reach_m=200)
    far = CameraLocation(wedge[len(wedge) // 2][0], wedge[len(wedge) // 2][1])
    assert 180 < GATE.distance_to(far) < 220


# ------------------------------------------------------------------ geofences


def test_a_geofence_needs_a_real_polygon():
    with pytest.raises(InvalidLocation):
        Geofence("TOO SMALL", ((28.0, 77.0), (28.1, 77.1)))


def test_a_geofence_point_off_the_earth_is_refused():
    with pytest.raises(InvalidLocation):
        Geofence("BAD", ((28.0, 77.0), (28.1, 77.1), (999.0, 77.2)))


def test_a_geofence_knows_what_is_inside_it():
    assert SECTOR.contains(28.6139, 77.2090)
    assert not SECTOR.contains(28.5000, 77.2090)
    assert not SECTOR.contains(28.6139, 77.3000)


def test_a_geofence_round_trips_through_json():
    restored = Geofence.from_dict(json.loads(json.dumps(SECTOR.to_dict())))
    assert restored == SECTOR


# ------------------------------------------------------------------- site map


def test_the_map_reports_which_areas_a_camera_stands_in():
    site = SiteMap({"CAM-01": GATE, "CAM-02": RIVER}, (SECTOR,))
    assert site.fences_around("CAM-01") == ["SECTOR-7"]
    assert site.fences_around("CAM-99") == []


def test_the_map_answers_which_camera_to_watch_next():
    """The question a map gets asked during an incident."""
    far = CameraLocation(28.7000, 77.3000)
    site = SiteMap({"CAM-01": GATE, "CAM-02": RIVER, "CAM-03": far})

    nearest = site.nearest("CAM-01")
    assert [camera for camera, _ in nearest] == ["CAM-02", "CAM-03"]
    assert nearest[0][1] < nearest[1][1]


def test_an_unlocated_camera_has_no_neighbours():
    assert SiteMap({"CAM-01": GATE}).nearest("CAM-99") == []


def test_the_map_payload_carries_everything_a_view_needs():
    payload = SiteMap({"CAM-01": GATE, "CAM-02": RIVER}, (SECTOR,)).to_dict()

    assert payload["located"] == 2
    first = payload["cameras"][0]
    assert first["camera_id"] == "CAM-01"
    assert first["site"] == "Gate A"
    assert first["coverage"], "a surveyed camera should carry its wedge"
    assert first["geofences"] == ["SECTOR-7"]
    assert payload["cameras"][1]["coverage"] == [], "unsurveyed bearing, no wedge"
    assert payload["geofences"][0]["name"] == "SECTOR-7"


def test_a_site_map_loads_from_a_file(tmp_path):
    path = tmp_path / "site.json"
    path.write_text(json.dumps({
        "cameras": {"CAM-01": {"lat": 28.6, "lon": 77.2, "site": "Gate A"}},
        "geofences": [SECTOR.to_dict()],
    }), encoding="utf-8")

    site = SiteMap.from_file(path)
    assert len(site) == 1
    assert site.locate("CAM-01").site == "Gate A"
    assert site.geofences[0].name == "SECTOR-7"


def test_a_missing_site_map_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        SiteMap.from_file(tmp_path / "nope.json")


# ------------------------------------------------------- the fleet and the API


def test_a_camera_spec_carries_its_location():
    from app.runner import CameraSpec

    spec = CameraSpec.from_dict({
        "camera_id": "CAM-01", "source": "webcam:0",
        "location": {"lat": 28.6, "lon": 77.2, "site": "Gate A"},
    })
    assert spec.location.site == "Gate A"


def test_a_camera_without_a_location_still_builds():
    """The common case. Nothing about running a camera depends on the map."""
    from app.runner import CameraSpec

    assert CameraSpec.from_dict({"camera_id": "CAM-01", "source": "webcam:0"}).location is None


def test_a_bad_location_names_the_camera_it_came_from():
    from app.runner import CameraSpec

    with pytest.raises(ValueError, match="CAM-07"):
        CameraSpec.from_dict({
            "camera_id": "CAM-07", "source": "webcam:0",
            "location": {"lat": 999, "lon": 77.2},
        })


def test_a_location_is_not_mistaken_for_a_config_override():
    """`location` is map data, not a SurveillanceConfig setting."""
    from app.config import SurveillanceConfig
    from app.runner import CameraSpec

    spec = CameraSpec.from_dict({
        "camera_id": "CAM-01", "source": "webcam:0",
        "location": {"lat": 28.6, "lon": 77.2},
    })
    assert "location" not in spec.overrides
    spec.build_config(SurveillanceConfig(source="synthetic"))   # must not raise


def test_the_shipped_fleet_file_loads_as_a_map():
    from app.runner import load_site_map

    site = load_site_map("config/cameras.json")
    assert len(site) >= 1
    assert site.geofences, "the example fleet should show what a geofence looks like"


def test_the_map_endpoint_serves_the_fleet(tmp_path):
    from fastapi.testclient import TestClient

    from app.config import SurveillanceConfig
    from app.runner import CameraSpec
    from app.server import create_app

    specs = [
        CameraSpec(camera_id="CAM-01", source="synthetic", location=GATE),
        CameraSpec(camera_id="CAM-02", source="synthetic"),      # not surveyed
    ]
    config = SurveillanceConfig(
        source="synthetic", detector="sim", synthetic_frames=3,
        zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, quiet=True, print_summary=False, color=False,
    )
    app = create_app(specs, config, db_path=tmp_path / "map.db", geofences=(SECTOR,))

    with TestClient(app) as client:
        payload = client.get("/api/map").json()
        assert [c["camera_id"] for c in payload["cameras"]] == ["CAM-01"]
        # Listed rather than dropped, so it is visible that it exists.
        assert payload["unplaced"] == ["CAM-02"]
        assert payload["geofences"][0]["name"] == "SECTOR-7"

        one = client.get("/api/map/cameras/CAM-01").json()
        assert one["site"] == "Gate A"
        assert one["geofences"] == ["SECTOR-7"]

        missing = client.get("/api/map/cameras/CAM-02")
        assert missing.status_code == 404
        assert "location" in missing.json()["detail"]


def test_the_running_pipeline_is_unaffected_by_the_map(tmp_path):
    """The map is an addition. Nothing about detection or alerting changes."""
    from app.config import SurveillanceConfig
    from app.pipeline import SurveillancePipeline

    config = SurveillanceConfig(
        source="synthetic", detector="sim", synthetic_frames=40, force_night=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, color=False, quiet=True, print_summary=False,
    )
    assert SurveillancePipeline(config).run().alerts == 1


def test_half_a_wedge_is_flagged_rather_than_silently_dropped(tmp_path, monkeypatch):
    """Someone who fills in a bearing and gets a bare pin deserves to be told why."""
    import json

    from app import ui
    from app.runner import load_site_map

    said: list[str] = []
    monkeypatch.setattr(ui, "warn", said.append)

    path = tmp_path / "fleet.json"
    path.write_text(json.dumps({"cameras": [
        {"camera_id": "CAM-A", "source": "x",
         "location": {"lat": 28.6, "lon": 77.2, "bearing": 90}},
        {"camera_id": "CAM-B", "source": "x",
         "location": {"lat": 28.6, "lon": 77.2, "fov": 70}},
        {"camera_id": "CAM-C", "source": "x",
         "location": {"lat": 28.6, "lon": 77.2, "bearing": 90, "fov": 70}},
        {"camera_id": "CAM-D", "source": "x",
         "location": {"lat": 28.6, "lon": 77.2}},
    ]}), encoding="utf-8")

    site = load_site_map(path)

    flagged = [w for w in said if "wedge" in w]
    assert len(flagged) == 2, f"expected CAM-A and CAM-B only, got {flagged}"
    assert any("CAM-A" in w and "fov is missing" in w for w in flagged)
    assert any("CAM-B" in w and "bearing is missing" in w for w in flagged)

    # A camera with neither is the normal unsurveyed case and says nothing.
    assert site.locate("CAM-D").coverage_wedge() == []
    assert site.locate("CAM-C").coverage_wedge(), "a complete pair must draw one"
