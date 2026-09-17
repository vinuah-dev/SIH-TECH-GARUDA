"""SENTINEL-X API server.

Runs the camera fleet in the background and exposes what it produces:

    GET  /                    operator dashboard
    GET  /api/status          per-camera state, link health, counters
    GET  /api/summary         totals by severity and camera
    GET  /api/events          recent events, filterable
    GET  /api/events/{id}     one event in full
    GET  /api/behaviours      how often each behaviour has fired
    GET  /api/incidents       events grouped into incidents
    GET  /api/map             where every camera is, and the areas around them
    PUT  /api/map/cameras/{id}     place or move a camera, saved to the fleet file
    DELETE /api/map/cameras/{id}   take a camera off the map
    WS   /ws                  live alerts, pushed as they happen

The pipeline knows nothing about any of this. It calls its event hooks, and
here the hook happens to be a WebSocket broadcast.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import (FastAPI, HTTPException, Query, Request, Response, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from .. import ui
from ..camera_sources import (ProbeFailed, check_camera_id, describe_source,
                              find_webcams, grab_frame, normalize_source,
                              preview_data_url, redact_source, webcam_index)
from ..config import SurveillanceConfig
from ..runner import (CameraSpec, MultiCameraRunner, load_camera_specs,
                      load_site_map)
from ..geo import DEFAULT_BASEMAPS, CameraLocation, InvalidLocation, SiteMap
from ..store import EventStore
from ..zones.manager import ZONE_KINDS, ZoneManager
from .hub import AlertHub

STATIC = Path(__file__).parent / "static"


class FleetService:
    """Owns the background thread the camera fleet runs in."""

    def __init__(
        self,
        specs: list[CameraSpec],
        config: SurveillanceConfig,
        store: EventStore,
        hub: AlertHub,
        geofences: tuple = (),
        fleet_path: Path | str | None = None,
        basemaps: tuple | None = None,
    ) -> None:
        self.specs = specs
        self.config = config
        self.store = store
        self.hub = hub
        # The file camera positions set on the map are saved into. Fixed here,
        # when the server starts - never taken from a request. None for a
        # single --source run, which has no fleet file to write to.
        self.fleet_path = Path(fleet_path) if fleet_path is not None else None
        self.fleet_lock = threading.Lock()
        # Built from the specs themselves, so a location lives beside the
        # camera it belongs to and there is no second file to keep in step.
        # Cameras with no surveyed coordinates simply do not appear in it.
        self.site_map = SiteMap(
            cameras={s.camera_id: s.location for s in specs if s.location is not None},
            geofences=geofences or (),
            basemaps=DEFAULT_BASEMAPS if basemaps is None else tuple(basemaps),
        )
        self.runner: MultiCameraRunner | None = None
        self.thread: threading.Thread | None = None
        self.started_at: datetime | None = None
        # Set once the fleet has finished probing and starting its cameras.
        # `runner` exists long before that, so waiting on the object alone
        # hands the caller an empty fleet.
        self.ready = threading.Event()

    def _publish(self, event) -> None:
        self.hub.publish_threadsafe({"type": "alert", "event": event.to_dict()})

    def start(self) -> None:
        self.runner = MultiCameraRunner(
            self.specs, self.config, store=self.store, event_hooks=[self._publish],
            # The dashboard streams the annotated view, so frames are kept even
            # when no desktop window is open.
            capture_frames=True,
        )
        self.started_at = datetime.now()
        try:
            # Build synchronously so /api/status and shutdown never race setup.
            started = self.runner.start()
        finally:
            self.ready.set()
        if not started:
            return
        if self.config.view:
            # OpenCV's GUI only works from the main thread, so when windows are
            # wanted the caller drives runner.wait() there instead. Waiting in
            # two places would put two threads back into the GUI.
            return
        self.thread = threading.Thread(target=self.runner.wait, daemon=True, name="sentinelx-fleet")
        self.thread.start()

    def stop(self) -> None:
        if self.runner is not None:
            self.runner.stop()
        if self.thread is not None:
            self.thread.join(timeout=10)

    @property
    def running(self) -> bool:
        # In view mode the wait loop lives on the caller's thread, so liveness
        # is whatever the cameras themselves say - and a camera added from the
        # dashboard after the wait loop ended still counts.
        cameras = bool(self.runner and any(w.alive for w in self.runner.workers))
        if self.thread is not None:
            return self.thread.is_alive() or cameras
        return cameras

    def status(self) -> dict:
        cameras = self.runner.status() if self.runner else []
        sources = {spec.camera_id: spec.source for spec in self.specs}
        return {
            "running": self.running,
            "started_at": self.started_at.isoformat(timespec="seconds")
            if self.started_at
            else None,
            "cameras": cameras,
            "skipped": [
                {"camera_id": cid, "reason": reason,
                 "source": redact_source(sources.get(cid, "")),
                 "label": describe_source(sources.get(cid, ""))}
                for cid, reason in (self.runner.skipped if self.runner else [])
            ],
            # Whether cameras can be added, removed and placed from the dashboard.
            "fleet_editable": self.fleet_path is not None,
            "model_sharing": "shared" if (self.runner and self.runner.shared) else "per camera",
            "total_alerts": sum(c["alerts"] for c in cameras),
            "dashboards_connected": self.hub.client_count,
        }


def create_app(
    specs: list[CameraSpec],
    config: SurveillanceConfig,
    db_path: Path | str = "data/sentinelx.db",
    geofences: tuple = (),
    fleet_path: Path | str | None = None,
    basemaps: tuple | None = None,
) -> FastAPI:
    store = EventStore(db_path)
    hub = AlertHub()
    service = FleetService(specs, config, store, hub, geofences=geofences,
                           fleet_path=fleet_path, basemaps=basemaps)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.bind(asyncio.get_running_loop())
        service.start()
        try:
            yield
        finally:
            service.stop()
            store.close()

    app = FastAPI(title="SENTINEL-X", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.state.store = store
    app.state.hub = hub

    # -------------------------------------------------------------- dashboard

    page_seen = {"mtime": None, "version": ""}

    def ui_version() -> str:
        """Short hash of the dashboard page, re-read only when the file changes."""
        page = STATIC / "index.html"
        try:
            mtime = page.stat().st_mtime_ns
        except OSError:
            return ""
        if mtime != page_seen["mtime"]:
            page_seen["version"] = hashlib.sha256(page.read_bytes()).hexdigest()[:12]
            page_seen["mtime"] = mtime
        return page_seen["version"]

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        page = STATIC / "index.html"
        if not page.exists():
            return HTMLResponse("<h1>SENTINEL-X</h1><p>Dashboard asset missing.</p>")
        # Never cached, and stamped with the version it is: a reload always
        # gets the current page, and an open tab can tell when it is behind.
        return HTMLResponse(
            page.read_text(encoding="utf-8").replace("__UI_VERSION__", ui_version()),
            headers={"Cache-Control": "no-store"},
        )

    # -------------------------------------------------------------------- api

    @app.get("/api/status")
    async def status() -> dict:
        return {**service.status(), "ui_version": ui_version()}

    @app.get("/api/summary")
    async def summary() -> dict:
        return store.summary()

    @app.get("/api/behaviours")
    async def behaviours() -> dict:
        return store.behaviour_counts()

    @app.get("/api/events")
    async def events(
        limit: int = Query(50, ge=1, le=500),
        camera: str | None = None,
        severity: str | None = None,
    ) -> dict:
        found = store.recent(limit=limit, camera_id=camera, severity=severity)
        return {"count": len(found), "events": found}

    @app.get("/api/incidents")
    async def incidents(
        limit: int = Query(50, ge=1, le=500),
        gap_seconds: float = Query(120.0, ge=1.0, le=3600.0),
        camera: str | None = None,
    ) -> dict:
        """One row per incident rather than one per alert.

        Four alerts saying the same person is still inside the fence is four
        chances to stop reading them.
        """
        found = store.incidents(limit=limit, gap_seconds=gap_seconds, camera_id=camera)
        return {"count": len(found), "incidents": found}

    # MJPEG multipart framing, spelled out once. The CRLF pairs are
    # load-bearing: get them wrong and the browser shows nothing at all,
    # with no error anywhere to say why.
    CRLF = bytes((13, 10))
    HEADER = b"--frame" + CRLF + b"Content-Type: image/jpeg" + CRLF + CRLF
    BOUNDARY_END = CRLF

    @app.get("/api/cameras/{camera_id}/stream")
    async def stream(camera_id: str):
        """The annotated view of one camera, as MJPEG.

        Always the newest frame, never a queue: a browser that falls behind
        should see live video, not catch up through a backlog. That is also
        why frames are re-encoded per request rather than shared - two viewers
        at different speeds must not stall each other.
        """
        import asyncio as _asyncio

        import cv2

        known = {spec.camera_id for spec in service.specs}
        if camera_id not in known:
            raise HTTPException(status_code=404, detail=f"no camera {camera_id}")

        async def frames():
            blank_sent = False
            while True:
                runner = service.runner
                frame = runner.latest_frame(camera_id) if runner else None
                if frame is None:
                    # Send one placeholder so the tile is not an empty box, then
                    # wait rather than spinning on a camera that is not running.
                    if not blank_sent:
                        blank_sent = True
                    await _asyncio.sleep(0.5)
                    continue
                blank_sent = False
                ok, buffer = cv2.imencode(".jpg", frame,
                                          [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    yield HEADER + buffer.tobytes() + BOUNDARY_END
                await _asyncio.sleep(0.08)

        return StreamingResponse(
            frames(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/api/integrity")
    async def integrity() -> dict:
        """Whether the event log still agrees with itself.

        Every event is hashed with the hash of the one before it, so an edited
        or deleted event breaks the chain and the break names the row. This is
        tamper-*evident*, not tamper-proof: publishing `head` somewhere the
        holder of this database does not control is what closes the gap, and
        nothing here does that for you.
        """
        report = store.verify_chain().to_dict()
        report["note"] = (
            "Tamper-evident, not tamper-proof. Publish the head hash off this "
            "machine - shift handover, district server, public chain - to make "
            "it a chain of custody."
        )
        return report

    @app.get("/api/cameras/{camera_id}/snapshot")
    async def snapshot(camera_id: str):
        """One still frame, for drawing fences on.

        A still rather than the live stream: the editor needs a picture that
        holds still while somebody clicks points onto it, and reusing the
        stream would restart it every time the page re-rendered.
        """
        import cv2

        if camera_id not in {spec.camera_id for spec in service.specs}:
            raise HTTPException(status_code=404, detail=f"no camera {camera_id}")

        runner = service.runner
        frame = runner.latest_frame(camera_id) if runner else None
        if frame is None:
            raise HTTPException(
                status_code=503,
                detail=f"{camera_id} has produced no frame yet - start the camera first",
            )
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise HTTPException(status_code=500, detail="could not encode the frame")
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    def _zone_file(camera_id: str) -> Path:
        """The fence file this camera actually uses.

        Resolved from the fleet's own configuration, never from anything the
        caller sends. This endpoint writes to disk on an API with no
        authentication, so the set of writable paths must be fixed by the
        configuration and not by a request.
        """
        for spec in service.specs:
            if spec.camera_id == camera_id:
                return Path(spec.zones_path or config.zones_path)
        raise HTTPException(status_code=404, detail=f"no camera {camera_id}")

    @app.get("/api/cameras/{camera_id}/zones")
    async def read_zones(camera_id: str) -> dict:
        """The fences drawn on this camera, in normalized 0-1 coordinates."""
        path = _zone_file(camera_id)
        if not path.exists():
            return {"camera_id": camera_id, "path": str(path), "zones": []}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail=f"{path}: {exc}") from exc
        return {
            "camera_id": camera_id,
            "path": str(path),
            "zones": raw.get("zones", []),
            "kinds": sorted(ZONE_KINDS),
        }

    @app.put("/api/cameras/{camera_id}/zones")
    async def write_zones(camera_id: str, payload: dict) -> dict:
        """Replace this camera's fences, and apply them without a restart.

        Validated by building the real ZoneManager from it *before* anything is
        written. A dashboard that can save a broken fence file is a dashboard
        that can stop a post detecting anything, and the next restart would be
        the first anyone heard of it.
        """
        path = _zone_file(camera_id)
        document = {
            "camera_id": camera_id,
            "_comment": payload.get("_comment", "Drawn in the SENTINEL-X dashboard."),
            "zones": payload.get("zones", []),
        }
        try:
            manager = ZoneManager.from_dict(document)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document, indent=2) + chr(10), encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"{path}: {exc}") from exc

        reloaded = 0
        for worker in (service.runner.workers if service.runner else []):
            if worker.spec.camera_id == camera_id:
                try:
                    reloaded = worker.pipeline.reload_zones()
                except (OSError, ValueError) as exc:
                    # Written but not applied. Saying which is the difference
                    # between "try again" and "restart to pick it up".
                    raise HTTPException(
                        status_code=500,
                        detail=f"saved to {path}, but the running camera could "
                               f"not reload it: {exc}",
                    ) from exc
        return {
            "camera_id": camera_id,
            "path": str(path),
            "saved": len(manager),
            "applied_live": bool(reloaded),
        }

    @app.get("/api/evidence/{name}")
    async def evidence(name: str):
        """One evidence file - a still, a clip or a plate crop.

        The name is looked up against the directory listing rather than joined
        onto it. Joining a caller-supplied path is how "../../etc/passwd"
        becomes a served file, and this endpoint sits on an API with no
        authentication at all, so it must not be the weak point.
        """
        directory = Path(config.evidence_dir)
        try:
            match = next(
                (p for p in directory.iterdir() if p.is_file() and p.name == name),
                None,
            )
        except OSError:
            match = None
        if match is None:
            raise HTTPException(status_code=404, detail=f"no evidence file {name}")
        return FileResponse(match)

    @app.get("/api/map")
    async def site_map() -> dict:
        """Where every camera is, and the areas drawn around them.

        The map is the answer to a question an alert cannot answer on its own:
        CAM-05 means nothing to somebody who does not already know where CAM-05
        is, but *this stretch of the river* tells a commander which patrol is
        nearest.

        Cameras without surveyed coordinates are listed separately rather than
        dropped, so it is visible that they exist and are simply not placed yet.
        """
        located = service.site_map.to_dict()
        placed = {camera["camera_id"] for camera in located["cameras"]}
        located["unplaced"] = [
            spec.camera_id for spec in service.specs if spec.camera_id not in placed
        ]
        # Whether positions set on the map can be saved. The dashboard says so
        # up front rather than letting somebody place six cameras and then
        # learn that none of it can be kept.
        located["editable"] = service.fleet_path is not None
        return located

    @app.get("/api/map/cameras/{camera_id}")
    async def camera_on_map(camera_id: str) -> dict:
        """One camera's place, and what is next to it.

        `nearest` exists for the question a map gets asked during an incident:
        somebody left CAM-02 heading east - which camera should be watched now?
        """
        where = service.site_map.locate(camera_id)
        if where is None:
            raise HTTPException(
                status_code=404,
                detail=f"no surveyed location for {camera_id}; add one under "
                       f"'location' in the fleet configuration",
            )
        return {
            "camera_id": camera_id,
            **where.to_dict(),
            "coverage": where.coverage_wedge(),
            "geofences": service.site_map.fences_around(camera_id),
            "nearest": [
                {"camera_id": other, "metres": round(distance, 1)}
                for other, distance in service.site_map.nearest(camera_id)
            ],
        }

    # ------------------------------------------------------ placing cameras

    def _spec(camera_id: str) -> CameraSpec:
        for spec in service.specs:
            if spec.camera_id == camera_id:
                return spec
        raise HTTPException(status_code=404, detail=f"no camera {camera_id}")

    def _fleet_file() -> Path:
        """The fleet file a camera position is saved into.

        Fixed when the server starts, from its --cameras argument, and never
        taken from a request. Like the zone editor this writes to disk on an API
        with no authentication, so which file gets written must not be
        something a caller can choose.
        """
        if service.fleet_path is None:
            raise HTTPException(
                status_code=409,
                detail="this server was started with a single --source, so there "
                       "is no fleet file to save a camera position into. Start it "
                       "with --cameras config/cameras.json instead.",
            )
        return service.fleet_path

    def _edit_fleet(change) -> Path:
        """Apply one change to the fleet file, all or nothing.

        The new file is written beside the old one and loaded back exactly as
        the server loads it at startup before it replaces anything, so a save
        that would stop the fleet starting next time never lands. Everything
        the change does not touch - notes, geofences, basemaps - is kept.
        """
        path = _fleet_file()
        with service.fleet_lock:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=500, detail=f"{path}: {exc}") from exc

            change(raw)

            staged = path.with_name(path.name + ".saving")
            try:
                staged.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + chr(10),
                                  encoding="utf-8")
                load_camera_specs(staged)
                load_site_map(staged)
                os.replace(staged, path)
            except (OSError, ValueError) as exc:
                staged.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=500, detail=f"{path} was left unchanged: {exc}",
                ) from exc
        return path

    def _write_location(camera_id: str, location: CameraLocation | None) -> Path:
        """Set one camera's `location` in the fleet file, and touch nothing else.

        Only the location of a camera already in the file is written. Placing a
        camera cannot add one, remove one, or change where its video comes from:
        that is the add-camera endpoint's job alone, and it checks the source
        it is given far more strictly than a map click needs to be.
        """
        def place(raw: dict) -> None:
            entry = next(
                (c for c in raw.get("cameras", [])
                 if str(c.get("camera_id")) == camera_id),
                None,
            )
            if entry is None:
                raise HTTPException(
                    status_code=409,
                    detail=f"{camera_id} is running but is no longer in "
                           f"{service.fleet_path} - the file was edited after the "
                           f"server started. Restart it, then place the camera again.",
                )
            if location is None:
                entry.pop("location", None)
            else:
                entry["location"] = {
                    key: value for key, value in location.to_dict().items()
                    if value not in (None, "")
                }

        return _edit_fleet(place)

    @app.put("/api/map/cameras/{camera_id}")
    async def place_camera(camera_id: str, payload: dict) -> dict:
        """Put a camera on the map, or move or re-aim it, from the dashboard.

        Checked with the same CameraLocation the startup loader uses, so a
        coordinate off the Earth or a bearing of 400 degrees is refused here
        rather than discovered at the next restart. Saved into the fleet file
        and applied to the running map at once - no restart.

        Only the fields a location has are read. Anything else in the body,
        `source` included, is ignored.
        """
        spec = _spec(camera_id)
        site = str(payload.get("site") or "").strip()[:60]
        fields = {key: payload.get(key) for key in ("lat", "lon", "bearing", "fov", "height_m")}
        try:
            located = CameraLocation.from_dict({**fields, "site": site})
        except (InvalidLocation, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if located is None:
            raise HTTPException(status_code=400, detail="a camera location needs lat and lon")

        path = _write_location(camera_id, located)
        # Memory changes only once the file has, so a failed save leaves the
        # map showing what is actually on disk.
        spec.location = located
        service.site_map.cameras[camera_id] = located

        result = {
            "camera_id": camera_id,
            **located.to_dict(),
            "coverage": located.coverage_wedge(),
            "geofences": service.site_map.fences_around(camera_id),
            "saved": True,
            "path": str(path),
        }
        if located.half_surveyed:
            missing = "fov" if located.field_of_view is None else "bearing"
            result["warning"] = (
                f"saved, but a direction needs both bearing and fov - {missing} is "
                f"missing, so the map shows a pin and no wedge"
            )
        return result

    @app.delete("/api/map/cameras/{camera_id}")
    async def unplace_camera(camera_id: str) -> dict:
        """Take a camera off the map. It keeps running; it is just unplaced."""
        spec = _spec(camera_id)
        path = _write_location(camera_id, None)
        spec.location = None
        service.site_map.cameras.pop(camera_id, None)
        return {"camera_id": camera_id, "placed": False, "saved": True, "path": str(path)}

    # ------------------------------------------------------ adding cameras

    def _from_this_dashboard(request: Request, body: bool = True) -> None:
        """Refuse a write that some other website's page sent through a browser.

        The API has no login, and these endpoints make the server connect to
        an address it is given. A page on another site cannot read what comes
        back, but it could still tell an operator's browser to add a camera
        pointing wherever it liked. Browsers name the page a request came from,
        and a request from this dashboard names this server.
        """
        origin = request.headers.get("origin")
        host = request.headers.get("host", "")
        if (origin is not None and urlsplit(origin).netloc != host) or \
                request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(status_code=403,
                                detail="refused: this request came from another website")
        if body:
            kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if kind != "application/json":
                raise HTTPException(status_code=415, detail="send the camera as JSON")

    def _webcam_owner(source: str) -> str | None:
        """Which configured camera already uses this webcam, if any."""
        index = webcam_index(source)
        if index is None:
            return None
        return next((spec.camera_id for spec in service.specs
                     if webcam_index(spec.source) == index), None)

    @app.post("/api/cameras/test")
    async def test_camera(request: Request, payload: dict) -> dict:
        """Open a camera, read one picture and close it - before it joins.

        Answers ok false with a reason rather than an error status: a camera
        that does not connect is the normal result of a typo, not a fault.
        """
        _from_this_dashboard(request)
        try:
            source = normalize_source(payload.get("source"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        owner = _webcam_owner(source)
        if owner:
            return {"ok": False,
                    "reason": f"webcam {webcam_index(source)} is already in use by {owner}"}
        try:
            image = await asyncio.wait_for(asyncio.to_thread(grab_frame, source), timeout=25)
        except ProbeFailed as exc:
            return {"ok": False, "reason": str(exc)}
        except asyncio.TimeoutError:
            return {"ok": False, "reason": "the camera did not answer within 25 seconds"}
        except Exception as exc:  # noqa: BLE001 - OpenCV fails in many shapes
            return {"ok": False, "reason": f"could not open it ({type(exc).__name__})"}
        height, width = image.shape[:2]
        return {"ok": True, "width": width, "height": height,
                "label": describe_source(source), "preview": preview_data_url(image)}

    @app.post("/api/cameras/webcams")
    async def list_webcams(request: Request) -> dict:
        """The webcam numbers with a camera behind them. Turns each on briefly."""
        _from_this_dashboard(request, body=False)
        busy = {webcam_index(spec.source): spec.camera_id for spec in service.specs
                if webcam_index(spec.source) is not None}
        return {"webcams": await asyncio.to_thread(find_webcams, busy)}

    @app.post("/api/cameras")
    async def add_camera(request: Request, payload: dict) -> dict:
        """Add a camera to the fleet from the dashboard, and start it at once.

        Saved into the fleet file the server started with, so it is still there
        after a restart. Only a camera name and a source are read, and the
        source must be a webcam number, a camera's rtsp/http address or the
        demo feed - never a file path.
        """
        _from_this_dashboard(request)
        try:
            camera_id = check_camera_id(payload.get("camera_id"))
            source = normalize_source(payload.get("source"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _fleet_file()
        if any(spec.camera_id == camera_id for spec in service.specs):
            raise HTTPException(status_code=409,
                                detail=f"there is already a camera called {camera_id}")
        owner = _webcam_owner(source)
        if owner:
            raise HTTPException(
                status_code=409,
                detail=f"webcam {webcam_index(source)} is already used by {owner}",
            )

        entry = {"camera_id": camera_id, "source": source}

        def append(raw: dict) -> None:
            cameras = raw.setdefault("cameras", [])
            if any(str(c.get("camera_id")) == camera_id for c in cameras):
                raise HTTPException(
                    status_code=409,
                    detail=f"{camera_id} is already in {service.fleet_path} - restart "
                           f"the server to load it",
                )
            cameras.append(entry)

        path = _edit_fleet(append)
        spec = CameraSpec.from_dict(entry)
        service.specs.append(spec)
        running, reason = False, "the fleet is not running"
        if service.runner is not None:
            try:
                running, reason = await asyncio.to_thread(service.runner.add_camera, spec)
            except ValueError as exc:
                reason = str(exc)
        return {
            "camera_id": camera_id,
            "label": describe_source(source),
            "source": redact_source(source),
            "saved": True,
            "running": running,
            "reason": reason or None,
            "path": str(path),
        }

    @app.delete("/api/cameras/{camera_id}")
    async def remove_camera(camera_id: str, request: Request) -> dict:
        """Stop a camera and delete it from the fleet file, map position included.

        Alerts and evidence it already produced stay: they are the record.
        """
        _from_this_dashboard(request, body=False)
        _spec(camera_id)
        _fleet_file()
        if len(service.specs) <= 1:
            raise HTTPException(
                status_code=409,
                detail="the fleet needs at least one camera - add the new one first, "
                       "then remove this one",
            )

        def drop(raw: dict) -> None:
            cameras = raw.get("cameras", [])
            kept = [c for c in cameras if str(c.get("camera_id")) != camera_id]
            if len(kept) == len(cameras):
                raise HTTPException(
                    status_code=409,
                    detail=f"{camera_id} is running but is not in {service.fleet_path} "
                           f"- the file was edited after the server started",
                )
            raw["cameras"] = kept

        path = _edit_fleet(drop)
        service.specs[:] = [spec for spec in service.specs if spec.camera_id != camera_id]
        service.site_map.cameras.pop(camera_id, None)
        stopped = False
        if service.runner is not None:
            stopped = await asyncio.to_thread(service.runner.remove_camera, camera_id)
        return {"camera_id": camera_id, "removed": True, "was_running": stopped,
                "saved": True, "path": str(path)}

    @app.get("/api/events/{event_id}")
    async def event(event_id: str) -> dict:
        found = store.get(event_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no event {event_id}")
        return found

    # -------------------------------------------------------------- websocket

    _feed_warned: list[bool] = []

    @app.websocket("/ws")
    async def live(websocket: WebSocket) -> None:
        await hub.connect(websocket)
        try:
            # Prime the dashboard so it is not blank until the next alert.
            await websocket.send_json(
                {
                    "type": "snapshot",
                    "status": service.status(),
                    "events": store.recent(limit=25),
                }
            )
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        except (ConnectionError, RuntimeError):
            # A browser tab closing mid-send does not always surface as a clean
            # WebSocketDisconnect. That is routine, and not worth a line.
            pass
        except Exception as exc:
            # Anything else means the feed itself is broken - an event that will
            # not serialise, most likely. Swallowing it leaves an operator
            # watching a blank dashboard with nothing to explain why, so say it
            # once rather than once per reconnect attempt.
            if not _feed_warned:
                _feed_warned.append(True)
                ui.warn(
                    f"live alert feed failed: {type(exc).__name__}: {exc}. "
                    f"The dashboard will not update. REST endpoints still work; "
                    f"try /api/events to see whether the store itself is healthy."
                )
        finally:
            hub.disconnect(websocket)

    return app
