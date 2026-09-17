"""Multi-camera runner.

One process, one model, many cameras.

Running a separate process per camera works, but pays for it twice: each
process loads its own copy of the weights, and they all fight for the same
CPU. A border post with eight cameras cannot afford eight copies of YOLO in
memory. This runner loads the detector once, shares it across camera threads
behind a lock, and writes every alert into a single event store.

Serialising inference is not the loss it looks like: PyTorch already spreads
one forward pass across the available cores, so two threads calling it at once
mostly contend rather than gain. Sharing costs a little latency per camera and
saves the memory of an entire extra model per feed.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import ui
from .alerts.events import IntrusionEvent
from .anpr import PlateLedger
from .tracking import PersonLedger
from .config import SurveillanceConfig
from .geo import (DEFAULT_BASEMAPS, Basemap, CameraLocation, Geofence,
                  InvalidLocation, SiteMap)
from .detection.base import Detection
from .pipeline import QUIT_KEYS, WINDOW_PREFIX, SessionStats, SurveillancePipeline
from .store import EventStore


@dataclass
class CameraSpec:
    """One camera's entry in the fleet configuration."""

    camera_id: str
    source: str
    zones_path: str | None = None
    # Where this camera physically stands. Optional, and normally absent: a
    # post gets coordinates when somebody walks it with a GPS, and everything
    # runs identically until then.
    location: CameraLocation | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "CameraSpec":
        known = {"camera_id", "source", "zones", "zones_path", "location"}
        if "camera_id" not in raw or "source" not in raw:
            raise ValueError("each camera needs a camera_id and a source")
        try:
            located = CameraLocation.from_dict(raw.get("location"))
        except InvalidLocation as exc:
            raise ValueError(f"camera {raw['camera_id']}: {exc}") from exc
        return cls(
            camera_id=str(raw["camera_id"]),
            source=str(raw["source"]),
            zones_path=raw.get("zones") or raw.get("zones_path"),
            location=located,
            overrides={k: v for k, v in raw.items() if k not in known and not k.startswith("_")},
        )

    def build_config(self, base: SurveillanceConfig) -> SurveillanceConfig:
        """Take the shared settings, then apply this camera's own."""
        # The fleet prints one combined summary, so members stay quiet.
        changes: dict[str, Any] = {
            "camera_id": self.camera_id,
            "source": self.source,
            "print_summary": False,
        }
        if self.zones_path:
            changes["zones_path"] = Path(self.zones_path)
        for key, value in self.overrides.items():
            if not hasattr(base, key):
                raise ValueError(f"camera {self.camera_id}: unknown setting {key!r}")
            changes[key] = value
        return replace(base, **changes)


def load_site_map(path: str | Path) -> SiteMap:
    """The map view of a fleet file: where each camera is, and what areas exist.

    Read from the same file the cameras are configured in, so a location lives
    next to the camera it belongs to and there is no second file to keep in
    step.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"camera configuration not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    cameras: dict[str, CameraLocation] = {}
    for entry in raw.get("cameras", []):
        try:
            located = CameraLocation.from_dict(entry.get("location"))
        except InvalidLocation as exc:
            raise ValueError(f"camera {entry.get('camera_id')}: {exc}") from exc
        if located is not None:
            # Half a wedge draws nothing, and saying nothing about it leaves
            # somebody who filled in a bearing wondering why the map is bare.
            if located.half_surveyed:
                missing = "fov" if located.field_of_view is None else "bearing"
                ui.warn(
                    f"{entry['camera_id']}: a coverage wedge needs both bearing "
                    f"and fov - {missing} is missing, so the map will show a pin "
                    f"only. A 2.8 mm lens is about 90 degrees, a 4 mm about 70."
                )
            cameras[str(entry["camera_id"])] = located

    fences = []
    for fence in raw.get("geofences", []):
        try:
            fences.append(Geofence.from_dict(fence))
        except (InvalidLocation, TypeError, ValueError) as exc:
            raise ValueError(f"geofence {fence.get('name')!r}: {exc}") from exc

    # Map backgrounds. Absent means the public defaults; an empty list means a
    # coordinate grid only, for a post that must fetch nothing from outside.
    basemaps = DEFAULT_BASEMAPS
    if raw.get("basemaps") is not None:
        try:
            basemaps = tuple(Basemap.from_dict(b) for b in raw["basemaps"])
        except (InvalidLocation, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"basemaps: {exc}") from exc
    return SiteMap(cameras=cameras, geofences=tuple(fences), basemaps=basemaps)


def load_camera_specs(path: str | Path) -> list[CameraSpec]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"camera configuration not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("cameras", [])
    if not entries:
        raise ValueError(f"{path} defines no cameras")

    specs = [CameraSpec.from_dict(entry) for entry in entries]
    seen = [s.camera_id for s in specs]
    duplicates = {cid for cid in seen if seen.count(cid) > 1}
    if duplicates:
        # Duplicate ids would make the event log ambiguous about which feed an
        # alert came from, which defeats the point of a multi-camera post.
        raise ValueError(f"duplicate camera ids: {', '.join(sorted(duplicates))}")
    return specs


class SharedDetector:
    """Wraps one detector so several camera threads can use it safely."""

    def __init__(self, detector: Any) -> None:
        self._detector = detector
        self._lock = threading.Lock()
        self.name = f"{getattr(detector, 'name', 'detector')} (shared)"
        self.calls = 0

    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        with self._lock:
            self.calls += 1
            return self._detector.detect(frame)


@dataclass
class CameraWorker:
    """One camera's pipeline, running on its own thread."""

    spec: CameraSpec
    pipeline: SurveillancePipeline
    thread: threading.Thread | None = None
    error: BaseException | None = None

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run, name=f"sentinelx-{self.spec.camera_id}", daemon=True
        )
        self.thread.start()

    def _run(self) -> None:
        try:
            self.pipeline.run()
        except BaseException as exc:  # noqa: BLE001 - one camera must not kill the post
            self.error = exc
            ui.error(f"[{self.spec.camera_id}] pipeline stopped: {exc}")

    @property
    def alive(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def status(self) -> dict:
        stats: SessionStats = self.pipeline.stats
        health = self.pipeline._source_health
        return {
            "camera_id": self.spec.camera_id,
            "source": self.spec.source,
            "running": self.alive,
            "frames": stats.frames,
            "detections": stats.detections,
            "alerts": stats.alerts,
            "fps": round(stats.fps, 1),
            "link": health.to_dict() if health is not None else None,
            "error": self._explain(stats),
        }

    def _explain(self, stats: SessionStats) -> str | None:
        """Why this camera is not running, in words an operator can act on.

        A camera that raises says so through the exception. A camera that opens
        and then delivers nothing - the usual shape of "another program already
        has this webcam" - used to end silently, leaving the dashboard showing
        STOPPED with no reason beside a camera that *did* explain itself. Two
        failures that look identical on screen and need completely different
        fixes is the worst kind of gap.
        """
        if self.error is not None:
            return str(self.error)
        if self.alive or stats.frames:
            return None
        return (
            f"opened but delivered no frames - another program may be using "
            f"{self.spec.source}, or it has no video to give"
        )


class MultiCameraRunner:
    """Runs every configured camera in one process, sharing one model."""

    def __init__(
        self,
        specs: Sequence[CameraSpec],
        base_config: SurveillanceConfig,
        store: EventStore | None = None,
        event_hooks: Sequence[Callable[[IntrusionEvent], None]] = (),
        capture_frames: bool = False,
    ) -> None:
        self.specs = list(specs)
        self.base_config = base_config
        self.store = store
        self.event_hooks = list(event_hooks)
        self.workers: list[CameraWorker] = []
        # One plate ledger for every camera: a vehicle passes several of them,
        # and any one may get the only readable angle.
        self.plate_ledger = PlateLedger()
        # And one person ledger, so a subject can be followed between cameras.
        self.person_ledger = PersonLedger()
        self.detector: SharedDetector | None = None
        self.shared = False
        self.skipped: list[tuple[str, str]] = []
        # Keep the newest annotated frame from each camera even when no window
        # is open, so the dashboard can stream it. Only the newest is held -
        # a viewer that falls behind should see live video, not a backlog.
        self.capture_frames = capture_frames
        # OpenCV's GUI must be driven from one thread, so cameras post their
        # annotated frames here and the main loop draws them.
        self._frames: dict[str, Any] = {}
        self._frame_lock = threading.Lock()

    # ------------------------------------------------------------------ setup

    def _make_detector(self, config: SurveillanceConfig):
        """Build one detector using the pipeline's own factory."""
        return SurveillancePipeline(config, store=None)._build_detector()

    def _build_shared_detector(self) -> SharedDetector | None:
        """Load the model once, but only if this backend can be shared.

        A backend with per-stream state (ByteTrack, the scripted simulator)
        must get one instance per camera, or the feeds corrupt each other.
        """
        detector = self._make_detector(self.base_config)
        if not getattr(detector, "shareable", False):
            self.shared = False
            return None
        self.shared = True
        return SharedDetector(detector)

    def _post_frame(self, camera_id: str, frame) -> None:
        with self._frame_lock:
            self._frames[camera_id] = frame

    def latest_frame(self, camera_id: str):
        """The newest annotated frame from one camera, or None."""
        with self._frame_lock:
            return self._frames.get(camera_id)

    def _reachable(self, spec: CameraSpec) -> tuple[bool, str]:
        """Skip a camera that is not answering rather than stalling the fleet."""
        from .video.stream import is_network_source, tcp_reachable

        if is_network_source(spec.source):
            if tcp_reachable(spec.source, timeout=3.0):
                return True, ""
            return False, "not answering on the network"

        if spec.source.lower().startswith("webcam") or spec.source.isdigit():
            import cv2

            index = spec.source.partition(":")[2]
            device = int(index) if index.isdigit() else (
                int(spec.source) if spec.source.isdigit() else 0
            )
            capture = cv2.VideoCapture(device)
            try:
                if capture.isOpened():
                    return True, ""
                return False, f"no camera device at index {device}"
            finally:
                capture.release()

        return True, ""  # files and synthetic sources fail loudly on their own

    def _usable_specs(self) -> list[CameraSpec]:
        usable = []
        for spec in self.specs:
            ok, reason = self._reachable(spec)
            if ok:
                usable.append(spec)
            else:
                self.skipped.append((spec.camera_id, reason))
                ui.warn(f"Skipping {spec.camera_id}: {reason}")
        return usable

    def _build_workers(self) -> list[CameraWorker]:
        workers = []
        for spec in self.specs:
            config = spec.build_config(self.base_config)
            detector = self.detector if self.shared else self._make_detector(config)
            pipeline = SurveillancePipeline(
                config,
                event_hooks=self.event_hooks,
                store=self.store,
                detector=detector,
                frame_sink=(self._post_frame
                            if config.view or self.capture_frames else None),
                plate_ledger=self.plate_ledger,
                person_ledger=self.person_ledger,
            )
            workers.append(CameraWorker(spec=spec, pipeline=pipeline))
        return workers

    # -------------------------------------------------------------------- run

    def _pump_windows(self) -> bool:
        """Draw the fleet in one window. False when the operator quits.

        One window per camera does not survive a real post: four cameras means
        four windows to find and keep on top, and the attention that costs is
        spent at exactly the wrong moment. Above one camera this is a wall.
        """
        import cv2

        from .video.mosaic import compose

        with self._frame_lock:
            snapshot = dict(self._frames)
        if not snapshot:
            return True

        # Every configured camera holds its place whether or not it has sent a
        # frame yet, so tiles do not renumber themselves as cameras connect.
        wall = compose(snapshot, [spec.camera_id for spec in self.specs])
        if wall is None:
            return True
        cv2.imshow(self._window_title(), wall)
        return (cv2.waitKey(1) & 0xFF) not in QUIT_KEYS

    def _window_title(self) -> str:
        if len(self.specs) == 1:
            return f"{WINDOW_PREFIX} - {self.specs[0].camera_id}"
        return f"{WINDOW_PREFIX} - {len(self.specs)} cameras"

    def start(self) -> list[CameraWorker]:
        """Probe, load the model and start every camera thread.

        Deliberately synchronous. A caller that backgrounds the *waiting* still
        gets a fully built fleet the moment this returns, so status queries and
        shutdown never race against setup.
        """
        ui.banner()
        ui.info(f"Fleet: {len(self.specs)} cameras configured")
        self.specs = self._usable_specs()
        if not self.specs:
            ui.error("No camera is reachable - nothing to run.")
            return []
        if self.skipped:
            ui.info(f"Running {len(self.specs)} of the configured cameras.")
        ui.info("Loading AI Vision Engine (shared across all cameras)...")
        self.detector = self._build_shared_detector()
        if self.shared and self.detector is not None:
            ui.ok(f"Model loaded once, shared by every camera: {self.detector.name}")
        else:
            ui.warn(
                "This detector keeps per-stream state, so each camera gets its own "
                "instance (sharing would mix the feeds up)."
            )

        self.workers = self._build_workers()
        for worker in self.workers:
            ui.info(f"   - {worker.spec.camera_id:<10} {worker.spec.source}")
            worker.start()
        print()
        return self.workers

    def run(self, poll: float = 0.5) -> list[CameraWorker]:
        """Start the fleet and block until every camera finishes."""
        if not self.start():
            return []
        return self.wait(poll=poll)

    def wait(self, poll: float = 0.5) -> list[CameraWorker]:
        viewing = self.base_config.view
        interval = min(poll, 0.03) if viewing else poll
        try:
            while any(worker.alive for worker in self.workers):
                if viewing and not self._pump_windows():
                    ui.warn("Live view closed by operator.")
                    self.stop()
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            print()
            ui.warn("Interrupt received - stopping every camera.")
            self.stop()
        finally:
            if self.base_config.view:
                try:
                    import cv2

                    cv2.destroyAllWindows()
                except Exception:
                    pass
            self._summary()
        return self.workers

    def stop(self) -> None:
        for worker in self.workers:
            worker.pipeline.request_stop()
        for worker in self.workers:
            if worker.thread is not None:
                worker.thread.join(timeout=10)

    def status(self) -> list[dict]:
        return [worker.status() for worker in self.workers]

    def _summary(self) -> None:
        ui.section("FLEET SUMMARY")
        total_alerts = 0
        for worker in self.workers:
            status = worker.status()
            total_alerts += status["alerts"]
            state = "ERROR" if status["error"] else ("RUNNING" if status["running"] else "STOPPED")
            print(
                ui.field(status["camera_id"], "", 11)
                + f"{state:<8} frames {status['frames']:<7} "
                f"alerts {status['alerts']:<4} {status['fps']:.1f} fps"
            )
            if status["error"]:
                print(f"             {status['error']}")
        print()
        print(ui.field("Cameras", str(len(self.workers))))
        if self.skipped:
            print(ui.field("Skipped", ", ".join(f"{c} ({r})" for c, r in self.skipped)))
        print(ui.field("Total alerts", str(total_alerts)))
        if self.detector is not None:
            print(ui.field("Inference calls", str(self.detector.calls)))
        print(ui.field("Model sharing", "shared" if self.shared else "one per camera"))
        plates = self.plate_ledger.known_plates()
        if plates or self.plate_ledger.handoffs:
            print(ui.field("Plates read", f"{len(plates)}"))
            if self.person_ledger.links:
                print(
                    ui.field(
                        "Cross-camera",
                        f"{self.person_ledger.links} person(s) followed between cameras",
                    )
                )
            if self.plate_ledger.handoffs:
                print(
                    ui.field(
                        "Cross-camera", f"{self.plate_ledger.handoffs} plate(s) handed between cameras"
                    )
                )
        if self.store is not None:
            print(ui.field("Event store", f"{self.store.path} ({self.store.count()} total)"))
        print()
        ui.ok("Fleet stopped.")
        print()
