"""Setting cameras on the map from the dashboard.

Coordinates used to be typed into the fleet file by hand, which in practice
meant they were not typed at all: the map showed nothing, and every camera sat
in the "not surveyed" list. Now a camera is placed by clicking the map or
pasting a coordinate, and the position is written back into that same file.

What these tests hold it to:

**Nothing lands on disk that the server could not load back.** The file is
written beside the old one, loaded exactly as startup loads it, and only then
swapped in. A dashboard that can save a fleet file the server then refuses to
start with is worse than no dashboard.

**The request chooses the position, never the file, and never the source.**
This writes to disk on an API with no authentication. The file is the one the
server was started with, and only a camera's `location` is touched - a body
that says `"source": "rtsp://somewhere"` changes nothing about where that
camera's video comes from.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import SurveillanceConfig
from app.runner import load_camera_specs, load_site_map
from app.server import create_app

FLEET = {
    "_comment": "kept verbatim across saves",
    "cameras": [
        {
            "camera_id": "CAM-01",
            "source": "synthetic",
            "zones": "config/zones.json",
            "location": {"lat": 28.613939, "lon": 77.209023, "bearing": 90,
                         "fov": 60, "site": "Gate A"},
        },
        {"camera_id": "CAM-02", "source": "synthetic", "_note": "not surveyed yet"},
    ],
    "geofences": [{
        "name": "SECTOR-7", "kind": "SECTOR",
        "polygon": [[28.61, 77.205], [28.618, 77.205], [28.618, 77.213], [28.61, 77.213]],
    }],
    "basemaps": [{"name": "Post tiles", "url": "http://10.0.0.5/tiles/{z}/{x}/{y}.png",
                  "attribution": "in-house", "max_zoom": 18}],
}

RIVER = {"lat": 28.614512, "lon": 77.20987, "bearing": 210, "fov": 70,
         "height_m": 6, "site": "River Bank"}


def config_for(tmp_path):
    return SurveillanceConfig(
        source="synthetic", detector="sim", synthetic_frames=5,
        zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, quiet=True, print_summary=False, color=False,
    )


@pytest.fixture
def fleet(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(FLEET, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def client(tmp_path, fleet):
    app = create_app(
        load_camera_specs(fleet), config_for(tmp_path), db_path=tmp_path / "m.db",
        geofences=load_site_map(fleet).geofences, fleet_path=fleet,
        basemaps=load_site_map(fleet).basemaps,
    )
    with TestClient(app) as ready:
        ready.fleet = fleet
        yield ready


def on_disk(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw, {c["camera_id"]: c for c in raw["cameras"]}


# ------------------------------------------------------------------ placing


def test_the_map_says_positions_can_be_saved(client):
    assert client.get("/api/map").json()["editable"] is True


def test_an_unplaced_camera_can_be_put_on_the_map(client):
    assert client.get("/api/map").json()["unplaced"] == ["CAM-02"]

    response = client.put("/api/map/cameras/CAM-02", json=RIVER)
    assert response.status_code == 200, response.text
    assert response.json()["saved"] is True

    payload = client.get("/api/map").json()
    assert payload["unplaced"] == [], "it should be placed at once, no restart"
    placed = {c["camera_id"]: c for c in payload["cameras"]}
    assert placed["CAM-02"]["site"] == "River Bank"
    assert placed["CAM-02"]["coverage"], "bearing and fov were given, so it has a wedge"


def test_the_position_is_written_into_the_fleet_file(client):
    client.put("/api/map/cameras/CAM-02", json=RIVER)
    _, cameras = on_disk(client.fleet)
    location = cameras["CAM-02"]["location"]
    assert location["lat"] == pytest.approx(28.614512)
    assert location["bearing"] == 210 and location["fov"] == 70
    assert location["site"] == "River Bank"


def test_the_saved_file_loads_the_way_the_server_loads_it_at_startup(client):
    """The real promise: the next restart comes up with the camera placed."""
    client.put("/api/map/cameras/CAM-02", json=RIVER)
    site = load_site_map(client.fleet)
    assert site.locate("CAM-02").site == "River Bank"
    assert [s.camera_id for s in load_camera_specs(client.fleet)] == ["CAM-01", "CAM-02"]


def test_moving_a_camera_leaves_everything_else_in_the_file_alone(client):
    before, cameras_before = on_disk(client.fleet)
    client.put("/api/map/cameras/CAM-01", json={"lat": 28.6150, "lon": 77.2101,
                                                "bearing": 45, "fov": 90, "site": "Gate B"})
    after, cameras_after = on_disk(client.fleet)

    assert after["_comment"] == before["_comment"]
    assert after["geofences"] == before["geofences"]
    assert after["basemaps"] == before["basemaps"]
    assert cameras_after["CAM-02"] == cameras_before["CAM-02"]
    moved = dict(cameras_after["CAM-01"])
    assert moved.pop("location")["site"] == "Gate B"
    original = dict(cameras_before["CAM-01"])
    original.pop("location")
    assert moved == original, "only the location of the moved camera may change"


def test_a_position_without_a_direction_draws_a_pin_and_no_wedge(client):
    """An unsurveyed bearing is left out rather than guessed."""
    response = client.put("/api/map/cameras/CAM-02", json={"lat": 28.6146, "lon": 77.2099})
    assert response.status_code == 200
    assert response.json()["coverage"] == []
    _, cameras = on_disk(client.fleet)
    assert "bearing" not in cameras["CAM-02"]["location"]


def test_a_360_camera_is_placed_without_a_bearing_and_draws_a_circle(client):
    response = client.put("/api/map/cameras/CAM-02",
                          json={"lat": 28.6146, "lon": 77.2099, "fov": 360, "site": "Mast"})
    assert response.status_code == 200
    body = response.json()
    assert "warning" not in body, "a 360-degree camera is not missing a bearing"
    assert len(body["coverage"]) == 37
    _, cameras = on_disk(client.fleet)
    assert cameras["CAM-02"]["location"]["fov"] == 360
    assert "bearing" not in cameras["CAM-02"]["location"]


def test_the_map_sends_the_basemaps_from_the_fleet_file(client):
    assert [b["name"] for b in client.get("/api/map").json()["basemaps"]] == ["Post tiles"]


def test_half_a_direction_is_saved_but_flagged(client):
    response = client.put("/api/map/cameras/CAM-02",
                          json={"lat": 28.6146, "lon": 77.2099, "bearing": 90})
    assert response.status_code == 200
    assert "fov" in response.json()["warning"]


def test_the_camera_knows_which_areas_it_now_stands_in(client):
    response = client.put("/api/map/cameras/CAM-02", json=RIVER).json()
    assert response["geofences"] == ["SECTOR-7"]


# ----------------------------------------------------------------- removing


def test_a_camera_can_be_taken_off_the_map(client):
    response = client.delete("/api/map/cameras/CAM-01")
    assert response.status_code == 200
    assert client.get("/api/map").json()["unplaced"] == ["CAM-01", "CAM-02"]
    _, cameras = on_disk(client.fleet)
    assert "location" not in cameras["CAM-01"]
    assert cameras["CAM-01"]["source"] == "synthetic", "it is unplaced, not removed"


# ---------------------------------------------------------------- refusing


@pytest.mark.parametrize("bad", [
    {"lat": 128.0, "lon": 77.2},                       # off the Earth
    {"lat": 28.6, "lon": 77.2, "bearing": 400},        # not a compass bearing
    {"lat": 28.6, "lon": 77.2, "fov": 0},              # no field of view at all
    {"lat": "north-ish", "lon": 77.2},                 # not a number
    {"lat": 28.6, "lon": 77.2, "bearing": "east"},     # not a number either
    {"lon": 77.2},                                     # half a coordinate
    {},                                                # nothing
])
def test_an_impossible_position_is_refused_and_nothing_is_written(client, bad):
    before = client.fleet.read_text(encoding="utf-8")
    response = client.put("/api/map/cameras/CAM-02", json=bad)
    assert response.status_code == 400
    assert client.fleet.read_text(encoding="utf-8") == before
    assert client.get("/api/map").json()["unplaced"] == ["CAM-02"]


def test_an_unknown_camera_is_a_404(client):
    before = client.fleet.read_text(encoding="utf-8")
    assert client.put("/api/map/cameras/NOPE", json=RIVER).status_code == 404
    assert client.delete("/api/map/cameras/NOPE").status_code == 404
    assert client.fleet.read_text(encoding="utf-8") == before


def test_a_camera_missing_from_the_file_is_not_quietly_added_back(client):
    """The file was edited after startup. Refuse, rather than write a guess."""
    raw, _ = on_disk(client.fleet)
    raw["cameras"] = [c for c in raw["cameras"] if c["camera_id"] != "CAM-02"]
    client.fleet.write_text(json.dumps(raw), encoding="utf-8")

    response = client.put("/api/map/cameras/CAM-02", json=RIVER)
    assert response.status_code == 409
    _, cameras = on_disk(client.fleet)
    assert "CAM-02" not in cameras


def test_no_scratch_file_is_left_behind(client):
    client.put("/api/map/cameras/CAM-02", json=RIVER)
    client.put("/api/map/cameras/CAM-02", json={"lat": 999, "lon": 0})
    leftovers = [p.name for p in client.fleet.parent.iterdir() if p.name.endswith(".saving")]
    assert not leftovers


# --------------------------------------------------------------- the boundary


def test_the_request_cannot_change_where_a_cameras_video_comes_from(client):
    """No authentication on this API. A position is all it may set."""
    client.put("/api/map/cameras/CAM-02",
               json={**RIVER, "source": "rtsp://attacker.example/stream",
                     "camera_id": "CAM-99", "zones": "../../etc/passwd"})
    _, cameras = on_disk(client.fleet)
    assert cameras["CAM-02"]["source"] == "synthetic"
    assert "zones" not in cameras["CAM-02"]
    assert set(cameras) == {"CAM-01", "CAM-02"}
    assert set(cameras["CAM-02"]["location"]) <= {"lat", "lon", "bearing", "fov",
                                                  "height_m", "site"}


def test_the_write_path_comes_from_the_fleet_not_the_request():
    import inspect

    import app.server.api as api

    source = inspect.getsource(api.create_app)
    marker = source.index("def _fleet_file")
    body = source[marker:source.index("def _write_location", marker)]
    assert "service.fleet_path" in body, "the file must be the one the server started with"
    assert "payload" not in body, "the request must not influence which file is written"


def test_a_long_site_name_is_cut_rather_than_written_whole(client):
    client.put("/api/map/cameras/CAM-02", json={**RIVER, "site": "x" * 5000})
    _, cameras = on_disk(client.fleet)
    assert len(cameras["CAM-02"]["location"]["site"]) <= 60


def test_without_a_fleet_file_nothing_pretends_to_save(tmp_path, fleet):
    """A single --source run has no file to write to, and says so."""
    app = create_app(load_camera_specs(fleet), config_for(tmp_path),
                     db_path=tmp_path / "n.db")
    with TestClient(app) as bare:
        assert bare.get("/api/map").json()["editable"] is False
        response = bare.put("/api/map/cameras/CAM-02", json=RIVER)
        assert response.status_code == 409
        assert "--cameras" in response.json()["detail"]
        assert bare.get("/api/map").json()["unplaced"] == ["CAM-02"]


# ------------------------------------------------------------------ the page


def test_the_map_page_carries_the_camera_placer(client):
    page = client.get("/").text
    assert 'id="view-mapPage"' in page
    for hook in ("mCanvas", "mSave", "mRemove", "mPaste", "/api/map/cameras/",
                 "mBase", "basemaps", "mMode"):
        assert hook in page, f"the map page never uses {hook}"


def test_serve_hands_the_fleet_file_to_the_server():
    """Otherwise every save from a normal `python serve.py` run would be refused."""
    import inspect

    import serve

    source = inspect.getsource(serve.main)
    assert "fleet_path=" in source


def test_the_no_hardware_fleet_opens_no_hardware():
    """The fleet a newcomer runs first must never switch a camera on."""
    specs = load_camera_specs("config/cameras-demo.json")
    assert specs and all(s.source == "synthetic" for s in specs)
    site = load_site_map("config/cameras-demo.json")
    placed = {s.camera_id for s in specs if site.locate(s.camera_id)}
    assert placed and placed != {s.camera_id for s in specs}, (
        "some cameras placed so the map shows something, one left for the placer"
    )
