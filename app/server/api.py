"""IBVAP API server.

Runs the camera fleet in the background and exposes what it produces:

    GET  /                    operator dashboard
    GET  /api/status          per-camera state, link health, counters
    GET  /api/summary         totals by severity and camera
    GET  /api/events          recent events, filterable
    GET  /api/events/{id}     one event in full
    GET  /api/behaviours      how often each behaviour has fired
    GET  /api/incidents       events grouped into incidents
    WS   /ws                  live alerts, pushed as they happen

The pipeline knows nothing about any of this. It calls its event hooks, and
here the hook happens to be a WebSocket broadcast.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .. import ui
from ..config import SurveillanceConfig
from ..runner import CameraSpec, MultiCameraRunner
from ..geo import SiteMap
from ..store import EventStore
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
    ) -> None:
        self.specs = specs
        self.config = config
        self.store = store
        self.hub = hub
        # Built from the specs themselves, so a location lives beside the
        # camera it belongs to and there is no second file to keep in step.
        # Cameras with no surveyed coordinates simply do not appear in it.
        self.site_map = SiteMap(
            cameras={s.camera_id: s.location for s in specs if s.location is not None},
            geofences=geofences or (),
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
            self.specs, self.config, store=self.store, event_hooks=[self._publish]
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
        self.thread = threading.Thread(target=self.runner.wait, daemon=True, name="ibvap-fleet")
        self.thread.start()

    def stop(self) -> None:
        if self.runner is not None:
            self.runner.stop()
        if self.thread is not None:
            self.thread.join(timeout=10)

    @property
    def running(self) -> bool:
        if self.thread is not None:
            return self.thread.is_alive()
        # In view mode the wait loop lives on the caller's thread, so liveness
        # is whatever the cameras themselves say.
        return bool(self.runner and any(w.alive for w in self.runner.workers))

    def status(self) -> dict:
        cameras = self.runner.status() if self.runner else []
        return {
            "running": self.running,
            "started_at": self.started_at.isoformat(timespec="seconds")
            if self.started_at
            else None,
            "cameras": cameras,
            "skipped": [
                {"camera_id": cid, "reason": reason}
                for cid, reason in (self.runner.skipped if self.runner else [])
            ],
            "model_sharing": "shared" if (self.runner and self.runner.shared) else "per camera",
            "total_alerts": sum(c["alerts"] for c in cameras),
            "dashboards_connected": self.hub.client_count,
        }


def create_app(
    specs: list[CameraSpec],
    config: SurveillanceConfig,
    db_path: Path | str = "data/ibvap.db",
    geofences: tuple = (),
) -> FastAPI:
    store = EventStore(db_path)
    hub = AlertHub()
    service = FleetService(specs, config, store, hub, geofences=geofences)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.bind(asyncio.get_running_loop())
        service.start()
        try:
            yield
        finally:
            service.stop()
            store.close()

    app = FastAPI(title="IBVAP", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.state.store = store
    app.state.hub = hub

    # -------------------------------------------------------------- dashboard

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        page = STATIC / "index.html"
        if not page.exists():
            return "<h1>IBVAP</h1><p>Dashboard asset missing.</p>"
        return page.read_text(encoding="utf-8")

    # -------------------------------------------------------------------- api

    @app.get("/api/status")
    async def status() -> dict:
        return service.status()

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
