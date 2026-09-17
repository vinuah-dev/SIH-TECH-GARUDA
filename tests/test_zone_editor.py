"""Drawing virtual fences from the dashboard.

Fences used to be hand-written JSON in normalized coordinates, which meant
nobody redrew them: moving a camera and then editing decimals by hand is the
kind of friction that leaves a post running fences that no longer match what
the camera sees.

Two things make this safe enough to expose, and both are what these tests are
about:

**Nothing is written until it loads.** The real ZoneManager is built from the
payload first. A dashboard that can save a broken fence file is a dashboard
that can stop a post detecting anything, and the next restart would be the
first anyone heard of it.

**The path comes from the fleet, never from the request.** This endpoint writes
to disk on an API with no authentication at all, so the set of writable files
is fixed by the configuration.
"""

import json
import shutil

import pytest
from fastapi.testclient import TestClient

from app.config import SurveillanceConfig
from app.runner import CameraSpec
from app.server import create_app

SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


@pytest.fixture
def client(tmp_path):
    zones = tmp_path / "zones-cam01.json"
    shutil.copy("config/zones-cam01.json", zones)
    config = SurveillanceConfig(
        source="synthetic", detector="sim", synthetic_frames=20,
        zones_path=str(zones),
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, quiet=True, print_summary=False, color=False,
    )
    specs = [CameraSpec(camera_id="CAM-01", source="synthetic", zones_path=str(zones))]
    app = create_app(specs, config, db_path=tmp_path / "z.db")
    with TestClient(app) as ready:
        ready.zones_path = zones
        yield ready


def fence(name="RED ZONE", kind="RESTRICTED", risk=70, polygon=None):
    return {
        "name": name, "kind": kind, "base_risk": risk,
        "polygon": polygon or SQUARE,
        "objects": ["PERSON", "VEHICLE"],
    }


# ------------------------------------------------------------------ reading


def test_the_fences_on_a_camera_can_be_read(client):
    """Asserted on shape, not on names.

    The fixture copies the shipped fence file, and that file is *designed* to
    be rewritten from the dashboard - so pinning the zone names here would
    break the moment somebody used the feature this tests.
    """
    payload = client.get("/api/cameras/CAM-01/zones").json()
    assert payload["camera_id"] == "CAM-01"
    assert payload["zones"], "the shipped camera should come with fences"
    for zone in payload["zones"]:
        assert zone["name"] and zone["kind"] in payload["kinds"]
        assert len(zone["polygon"]) >= 3


def test_the_editor_is_told_which_kinds_exist(client):
    """So the dropdown cannot offer a kind the loader would then reject."""
    kinds = client.get("/api/cameras/CAM-01/zones").json()["kinds"]
    assert set(kinds) == {"RESTRICTED", "WATCH", "PATROL"}


def test_an_unknown_camera_is_a_404(client):
    assert client.get("/api/cameras/NOPE/zones").status_code == 404


# ------------------------------------------------------------------ writing


def test_a_drawn_fence_is_saved_and_read_back(client):
    response = client.put("/api/cameras/CAM-01/zones", json={"zones": [fence()]})
    assert response.status_code == 200
    assert response.json()["saved"] == 1

    saved = client.get("/api/cameras/CAM-01/zones").json()["zones"]
    assert [z["name"] for z in saved] == ["RED ZONE"]
    assert saved[0]["polygon"] == SQUARE


def test_a_saved_fence_reaches_the_running_camera(client):
    """Otherwise drawing a fence means restarting the post to use it."""
    response = client.put("/api/cameras/CAM-01/zones", json={"zones": [fence()]})
    assert response.json()["applied_live"] is True

    pipeline = client.app.state.service.runner.workers[0].pipeline
    assert [z.name for z in pipeline.zones] == ["RED ZONE"]
    # The context engine holds its own reference; left stale it would go on
    # tracking approach and dwell against a fence that no longer exists.
    assert pipeline.context.zones is pipeline.zones


def test_the_file_is_written_where_the_fleet_says(client):
    client.put("/api/cameras/CAM-01/zones", json={"zones": [fence()]})
    written = json.loads(client.zones_path.read_text(encoding="utf-8"))
    assert written["camera_id"] == "CAM-01"
    assert written["zones"][0]["name"] == "RED ZONE"


# ------------------------------------------------------------- refusing bad


def test_an_unknown_kind_is_refused(client):
    response = client.put("/api/cameras/CAM-01/zones",
                          json={"zones": [fence(kind="NONSENSE")]})
    assert response.status_code == 400
    assert "NONSENSE" in response.json()["detail"]


def test_a_polygon_that_is_not_a_shape_is_refused(client):
    response = client.put("/api/cameras/CAM-01/zones",
                          json={"zones": [fence(polygon=[[0.1, 0.1], [0.9, 0.9]])]})
    assert response.status_code == 400


def test_saving_no_fences_at_all_is_refused(client):
    """A camera with no fences detects nothing, which is never what was meant."""
    assert client.put("/api/cameras/CAM-01/zones", json={"zones": []}).status_code == 400


@pytest.mark.parametrize("bad", [
    {"zones": [{"kind": "WATCH", "polygon": SQUARE}]},                 # no name
    {"zones": [{"name": "X", "kind": "WATCH"}]},                       # no polygon
    {"zones": [{"name": "X", "kind": "WATCH", "polygon": "square"}]},  # not points
])
def test_a_malformed_fence_is_refused(client, bad):
    assert client.put("/api/cameras/CAM-01/zones", json=bad).status_code == 400


def test_a_refused_save_leaves_the_previous_fences_untouched(client):
    """Validation before writing is the whole point: a rejection changes nothing."""
    before = client.zones_path.read_text(encoding="utf-8")

    pipeline = client.app.state.service.runner.workers[0].pipeline
    running = [z.name for z in pipeline.zones]

    client.put("/api/cameras/CAM-01/zones", json={"zones": [fence(kind="NONSENSE")]})

    assert client.zones_path.read_text(encoding="utf-8") == before
    assert [z.name for z in pipeline.zones] == running, (
        "a rejected save must leave the running camera exactly as it was"
    )


def test_writing_to_an_unknown_camera_is_a_404(client):
    response = client.put("/api/cameras/NOPE/zones", json={"zones": [fence()]})
    assert response.status_code == 404


def test_the_write_path_comes_from_the_fleet_not_the_request(client):
    """No request field may choose the file. There is no auth on this API."""
    import inspect

    import app.server.api as api

    source = inspect.getsource(api.create_app)
    marker = source.index("def _zone_file")
    body = source[marker:source.index("@app.get", marker)]
    assert "service.specs" in body, "the path must be resolved from the fleet"
    assert "payload" not in body, "the request must not influence the path"


# ---------------------------------------------------------------- the page


def test_the_dashboard_carries_a_zone_editor(client):
    page = client.get("/").text
    assert "zonesPage" in page, "there is no way to reach the editor"
    for hook in ("/zones", "/snapshot", "zCanvas", "zSave"):
        assert hook in page, f"the editor never uses {hook}"


def test_the_editor_offers_only_kinds_the_loader_accepts(client):
    from app.zones.manager import ZONE_KINDS

    page = client.get("/").text
    for kind in ZONE_KINDS:
        assert f'value="{kind}"' in page, f"{kind} cannot be chosen"


def test_every_page_in_the_nav_has_a_view(client):
    """A nav entry with no view behind it is a dead link in a demo."""
    import re

    page = client.get("/").text
    wanted = set(re.findall(r'<a[^>]*data-view="(\w+)"', page))
    assert wanted, "the nav has no pages"
    for name in wanted:
        assert f'id="view-{name}"' in page, f"nav offers {name} with nothing behind it"
