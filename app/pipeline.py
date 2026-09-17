"""The SENTINEL-X surveillance pipeline.

    frame -> detection -> tracking -> zone check -> risk -> alert -> log

Each stage lives in its own module; this file only wires them together and
owns the console/session lifecycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from . import ui
from .alerts.console import render_alert, render_detection, render_status
from .alerts.events import IntrusionEvent, IntrusionMonitor, new_event_id
from .alerts.clips import ClipRecorder
from .anpr import ANPREngine
from .anpr.detector import PlateDetector
from .anpr.plate_log import PlateLog
from .faces import FaceEngine, FaceRecognitionEngine, FaceRoster
from .registry import VehicleRegistry
from .tracking import PersonLedger
from .alerts.sink import EventLog, EvidenceStore
from .behaviour.camera import CameraIntegrityMonitor
from .behaviour.engine import CAMERA_TAMPERING, Behaviour, BehaviourEngine
from .config import SurveillanceConfig
from .context.engine import ContextEngine, TrackContext
from .context.scene import SceneContext
from .detection.base import Detection
from .detection.classes import is_person, is_vehicle
from .detection.tracker import IoUTracker
from .risk.engine import RiskAssessment, RiskEngine
from .store import EventStore
from .video.annotate import annotate
from .video.sources import open_source
from .zones.geometry import normalize
from .zones.manager import Zone, ZoneManager

EVENT_TYPE = "VIRTUAL FENCE INTRUSION"
TAMPER_EVENT_TYPE = "CAMERA TAMPERING"
STATUS_INTERVAL = 2.0
WINDOW_PREFIX = "SENTINEL-X - Live Surveillance"
QUIT_KEYS = (ord("q"), 27)


@dataclass
class SessionStats:
    frames: int = 0
    detections: int = 0
    alerts: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return max(1e-6, time.time() - self.started_at)

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed


class SurveillancePipeline:
    """Wires ingestion, detection, zones, risk and alerting into one loop."""

    def __init__(
        self,
        config: SurveillanceConfig,
        event_hooks: "list[Callable[[IntrusionEvent], None]] | None" = None,
        store: EventStore | None = None,
        detector=None,
        frame_sink=None,
        plate_ledger=None,
        person_ledger=None,
    ) -> None:
        self.config = config
        # Hooks let a runner or API server observe alerts without the pipeline
        # knowing anything about them.
        self.event_hooks = list(event_hooks or [])
        self.store = store
        self.zones = ZoneManager.from_file(config.zones_path)
        self.context = ContextEngine(self.zones, armed_kinds=config.alert_kinds)
        self.behaviour = BehaviourEngine()
        self.integrity = CameraIntegrityMonitor()
        self.anpr = ANPREngine(
            enabled=config.anpr,
            camera_id=config.camera_id,
            ledger=plate_ledger,
            # Reading a plate costs 0.22 s with the default recogniser and
            # 0.56 s with EasyOCR, against roughly 100 ms for a frame. The
            # frame loop must not wait for it.
            background=config.anpr_background,
            ocr_backend=config.ocr_backend,
            plate_detector=(
                PlateDetector(config.plate_weights, device=config.device,
                              rescue_crops=config.plate_rescue)
                if config.anpr
                else None
            ),
        )
        self.registry = (
            VehicleRegistry.from_file(config.registry_path)
            if config.registry_path
            else VehicleRegistry()
        )
        # One check per track: the plate does not change while we watch it.
        self._registry_checks: dict[int, object] = {}
        self.identities = self._build_face_engine(config)
        # Shared by the fleet so one person can be followed between cameras.
        self.people = person_ledger
        self.risk = RiskEngine()
        self.monitor = IntrusionMonitor(
            confirm_frames=config.confirm_frames,
            cooldown_seconds=config.cooldown_seconds,
            alert_kinds=config.alert_kinds,
        )
        self.tracker = (
            IoUTracker(reid_enabled=config.reid) if config.tracker == "iou" else None
        )
        self.event_log = EventLog(config.events_dir)
        self.evidence = EvidenceStore(config.evidence_dir, enabled=config.save_evidence)
        self.passes = PlateLog(
            directory=config.events_dir,
            evidence=self.evidence if config.save_evidence else None,
            window=config.plate_log_window,
            enabled=config.plate_log and config.anpr,
        )
        self.clips = ClipRecorder(
            config.evidence_dir,
            enabled=config.save_clips,
            pre_seconds=config.clip_pre_seconds,
            post_seconds=config.clip_post_seconds,
        )
        self.stats = SessionStats()
        self.detector = detector
        # When set, annotated frames go here instead of to an OpenCV window.
        # OpenCV's GUI is not thread-safe: two camera threads both calling
        # waitKey deadlock the process, so a fleet displays on its main thread.
        self.frame_sink = frame_sink
        self._stop_requested = False
        self._source = None
        self._zone_of_track: dict[int, tuple[str, tuple[str, ...]]] = {}
        self._source_health = None
        self._stream_time = 0.0
        # One window per camera, or a fleet would draw every feed into one.
        self._window = f"{WINDOW_PREFIX} - {config.camera_id}"
        self._last_status = 0.0
        self._confirmation_warned = False
        # Above this, the confirmation window is long enough to miss a crossing.
        self.confirmation_warn_seconds = 1.5

    # ---------------------------------------------------------------- setup

    def _build_detector(self):
        if self.config.detector == "sim":
            from .detection.simulator import SimulatedPersonDetector

            return SimulatedPersonDetector(
                steps=self.config.synthetic_frames,
                walk_fraction=self.config.sim_walk_fraction,
                approach_camera_to=self.config.sim_approach_camera,
            )

        from .detection.yolo import YoloPersonDetector

        return YoloPersonDetector(
            weights=self.config.weights,
            confidence=self.config.confidence,
            imgsz=self.config.imgsz,
            device=self.config.device,
            use_bytetrack=self.config.tracker == "bytetrack",
            labels=self.config.labels,
        )

    @staticmethod
    def _build_face_engine(config: SurveillanceConfig) -> FaceRecognitionEngine:
        """Recognition only exists if a post has said who it expects to see."""
        if not config.faces_path:
            return FaceRecognitionEngine(enabled=False)
        return FaceRecognitionEngine(
            faces=FaceEngine(),
            roster=FaceRoster.from_file(config.faces_path),
        )

    def _relax_confirmation_for_stills(self, source) -> None:
        """A single still can never accumulate N consecutive in-zone frames.

        Without this, `--source photo.jpg` would silently never alert.
        """
        if source.kind != "IMAGE" or self.config.confirm_frames <= 1:
            return
        ui.warn(
            f"Still image source: confirmation lowered from "
            f"{self.config.confirm_frames} frames to 1."
        )
        self.config.confirm_frames = 1
        self.monitor.confirm_frames = 1

    def _startup(self, source) -> None:
        ui.info(f"Camera {self.config.camera_id} connected  [{source.kind}: {source.name}]")
        if getattr(source, "health", None) is not None:
            source.on_state_change = self._on_camera_state
        ui.info(f"AI Vision Engine initialized  [{self.detector.name}]")
        reid = "with re-ID" if self.config.reid else "no re-ID"
        ui.info(f"Tracker: {self.config.tracker.upper()} ({reid})")
        ui.info(f"Watching: {', '.join(self.config.labels)}")
        if self.config.anpr and any(
            is_vehicle(label) for label in self.config.labels
        ):
            backend = (self.anpr.reader_name if self.anpr.ocr_available
                       else "localisation only (no OCR)")
            detector = self.anpr.plate_detector
            if detector is not None and detector.available:
                finder = f"plate model ({detector.weights.name})"
            else:
                finder = "no plate model - falling back to whole-vehicle text search"
            mode = "background reader" if self.anpr.background else "inline"
            ui.info(f"ANPR: enabled - {finder}, {backend}, {mode}")
            if self.registry.enabled:
                ui.info(
                    f"Vehicle registry: {self.config.registry_path} "
                    f"({len(self.registry.backend)} records) - local extract, not VAHAN"
                )
        if self.identities.available:
            ui.info(
                f"Face roster: {self.config.faces_path} "
                f"({len(self.identities.roster)} enrolled) - recognition suppresses "
                f"alerts for expected people"
            )
        elif self.config.faces_path:
            reason = self.identities.faces.last_error if self.identities.faces else "no roster"
            ui.warn(f"Face recognition unavailable: {reason}")
        ui.info(f"Virtual fences loaded from {self.config.zones_path}")
        for zone in self.zones:
            marker = "ALERTING" if zone.kind in self.config.alert_kinds else "OBSERVE "
            ui.info(f"   - {zone.name:<12} {zone.kind:<11} base risk {zone.base_risk:<3} [{marker}]")
        ui.info(
            f"Alert policy: confirm {self.config.confirm_frames} frames, "
            f"cooldown {self.config.cooldown_seconds:.0f}s"
        )
        if self.store is not None:
            ui.info(f"Event store: {self.store.path} ({self.store.count()} events on record)")
        ui.ok("Surveillance started. Press Ctrl+C to stop.")
        print()

    # ------------------------------------------------------------- per frame

    def _on_camera_state(self, health) -> None:
        """Surface link changes, and raise a security alert when they warrant it.

        A camera that stops answering may have been tampered with, so the
        integrity monitor decides which transitions are noise and which are
        events worth an operator's attention.
        """
        message = f"Camera link {health.state.value}"
        if health.last_error:
            message += f" - {health.last_error}"
        if health.state.value in ("RECONNECTING", "OFFLINE"):
            ui.warn(message)
        else:
            ui.info(message)

        behaviours = self.integrity.observe(health.state, time.time())
        if behaviours:
            self._raise_camera_alert(behaviours)

    def _raise_camera_alert(self, behaviours: list[Behaviour]) -> IntrusionEvent:
        """An alert with no person in it: the subject is the camera itself."""
        context = SceneContext.build(
            self.config.camera_id,
            night_start=self.config.night_start,
            night_end=self.config.night_end,
            force_night=self.config.force_night,
        )
        assessment = self.risk.assess(None, None, context, None, behaviours)
        event_id = new_event_id()
        event = IntrusionEvent(
            event_id=event_id,
            timestamp=context.timestamp,
            camera_id=self.config.camera_id,
            event_type=behaviours[0].name,
            detection=None,
            zone=None,
            risk=assessment,
            frame_index=self.stats.frames,
            # There is no frame to keep: the camera is what failed.
            evidence_path=None,
            track_context=None,
            behaviours=list(behaviours),
        )
        self.event_log.write(event)
        if self.store is not None:
            self.store.write(event)
        if not self.config.quiet:
            print(render_alert(event))
        self.stats.alerts += 1
        self._notify(event)
        return event

    def _locate(self, detection: Detection, width: int, height: int) -> Zone | None:
        point = normalize(detection.ground_point, width, height)
        zone = self.zones.locate(point)
        # A vehicle gate is a fence for people but not for the trucks it exists
        # to admit, so a zone that does not cover this object kind is not one.
        if zone is not None and not zone.applies_to(detection.label):
            return None
        return zone

    def _observe_context(
        self,
        detection: Detection,
        zone: Zone | None,
        scene: SceneContext,
        width: int,
        height: int,
        now: float,
    ) -> TrackContext | None:
        """Feed the Context Engine. Untracked detections have no history."""
        if detection.track_id is None:
            return None
        x1, y1, x2, y2 = detection.bbox
        return self.context.observe(
            detection.track_id,
            normalize(detection.ground_point, width, height),
            zone,
            scene,
            now,
            bbox_norm=(x1 / width, y1 / height, x2 / width, y2 / height),
        )

    def _raise_alert(
        self, detection, zone, context, frame_index, annotated,
        track=None, assessment=None, behaviours=None, event_type=EVENT_TYPE,
    ) -> IntrusionEvent:
        if assessment is None:
            assessment = self.risk.assess(detection, zone, context, track, behaviours)
        event_id = new_event_id()
        evidence_path = self.evidence.save(
            annotated, event_id, context.timestamp, self.config.camera_id
        )
        clip_path = self.clips.start(
            event_id, self.config.camera_id, context.timestamp, self._stream_time
        )
        plate, plate_path = self._plate_evidence(detection, event_id, context.timestamp)
        check = (
            self._registry_checks.get(detection.track_id)
            if detection is not None and detection.track_id is not None
            else None
        )
        event = IntrusionEvent(
            event_id=event_id,
            timestamp=context.timestamp,
            camera_id=self.config.camera_id,
            event_type=event_type,
            detection=detection,
            zone=zone,
            risk=assessment,
            frame_index=frame_index,
            evidence_path=evidence_path,
            clip_path=str(clip_path) if clip_path else None,
            plate_path=plate_path,
            plate=plate,
            registry=check.to_dict() if check is not None else None,
            track_context=track,
            subject=(
                subject.to_dict()
                if detection is not None and self.people is not None
                and (subject := self.people.subject_for(
                    self.config.camera_id, detection.track_id))
                else None
            ),
            identity=(
                match.to_dict()
                if detection is not None
                and (match := self.identities.identity_for(detection.track_id))
                else None
            ),
            behaviours=list(behaviours or []),
        )
        self.event_log.write(event)
        if self.store is not None:
            self.store.write(event)
        if not self.config.quiet:
            ui.emit(render_alert(event))
        self.stats.alerts += 1
        self._notify(event)
        return event

    def _notify(self, event: IntrusionEvent) -> None:
        """Fan the event out to observers. One bad hook must not stop the feed."""
        for hook in self.event_hooks:
            try:
                hook(event)
            except Exception as exc:  # pragma: no cover - defensive
                ui.warn(f"event hook failed: {exc}")

    def _verify_plate(self, detection):
        """Check a vehicle's plate against the registry, once per track."""
        if not self.registry.enabled or detection.track_id is None:
            return None
        if detection.track_id in self._registry_checks:
            return self._registry_checks[detection.track_id]

        reading = self.anpr.best_for(detection.track_id)
        plate = reading.text if reading else None
        if plate is None:
            # Another camera in the fleet may have read what this one could not.
            handed = self.anpr.fleet_plate(detection.track_id)
            plate = handed.text if handed else None
        if plate is None:
            return None  # nothing to check yet; try again next frame

        check = self.registry.check(plate, detection.label)
        self._registry_checks[detection.track_id] = check
        if check is not None and (check.mismatch or check.flagged):
            ui.warn(f"[{self.config.camera_id}] {plate}: {check.detail}")
        return check

    def _follow_across_cameras(self, detection: Detection, frame, now: float) -> None:
        """Tell the fleet ledger what this camera can see of this person."""
        if self.people is None or detection.track_id is None:
            return
        if not is_person(detection.label):
            return
        from .detection.appearance import describe

        self.people.observe(
            self.config.camera_id,
            detection.track_id,
            now,
            describe(frame, detection.bbox),
        )

    def _plate_evidence(self, detection, event_id, when):
        """The plate number and its crop, for a vehicle that has triggered an alert."""
        if detection is None or detection.track_id is None:
            return None, None
        reading = self.anpr.best_for(detection.track_id)
        crop = self.anpr.crop_for(detection.track_id)
        fleet = self.anpr.fleet_plate(detection.track_id)

        path = None
        if crop is not None and self.config.save_evidence:
            saved = self.evidence.save(
                crop, f"{event_id}-plate", when, self.config.camera_id
            )
            path = saved
        payload = reading.to_dict() if reading else None
        if payload is None and fleet is not None:
            # Read by another camera and handed to this one; say so plainly.
            payload = {
                "text": fleet.text,
                "display": fleet.text,
                "state": None,
                "confidence": fleet.confidence,
                "reads": fleet.reads,
                "bbox": [],
                "raw": "",
            }
        if payload is not None and fleet is not None:
            payload["fleet"] = fleet.to_dict()
        return payload, path

    def _should_log_detection(
        self, detection: Detection, zone: Zone | None, behaviours: list[Behaviour] | None = None
    ) -> bool:
        """Log on state change only - a line per person per frame is noise.

        A newly detected behaviour counts as a state change: an operator wants
        to see LOITERING appear the moment it is recognised.
        """
        if detection.track_id is None:
            return True
        current = zone.name if zone else "OPEN"
        names = tuple(sorted(b.name for b in behaviours or []))
        state = (current, names)
        changed = self._zone_of_track.get(detection.track_id) != state
        self._zone_of_track[detection.track_id] = state
        return changed or self.config.verbose

    # ------------------------------------------------------------------ run

    def run(self) -> SessionStats:
        ui.banner()
        source = open_source(
            self.config.source,
            stride=self.config.stride,
            limit=self.config.limit,
            loop=self.config.loop,
            max_width=self.config.max_width,
            synthetic_frames=self.config.synthetic_frames,
            reconnect_attempts=self.config.reconnect_attempts,
        )
        self._source = source
        self._source_health = getattr(source, "health", None)
        try:
            if self.detector is None:
                ui.info("Loading AI Vision Engine...")
                self.detector = self._build_detector()
            self._relax_confirmation_for_stills(source)
            self._startup(source)
            self.anpr.start()
            self._loop(source)
        except KeyboardInterrupt:
            print()
            ui.warn("Interrupt received - shutting down.")
        finally:
            source.release()
            self._teardown()
        return self.stats

    def request_stop(self) -> None:
        """Ask the run loop to finish after the current frame."""
        self._stop_requested = True
        # A network camera that is down never delivers a next frame, so the
        # flag above would never be read; tell the source itself to give up.
        stop = getattr(self._source, "stop", None)
        if callable(stop):
            stop()

    def _loop(self, source) -> None:
        import cv2

        # Model loading already happened; time only the streaming work so the
        # reported FPS reflects real throughput.
        self.stats.started_at = time.time()
        for frame in source.frames():
            if self._stop_requested:
                break
            self.stats.frames += 1
            width, height = frame.size
            # Speed, dwell and cooldown run on footage time; the day/night
            # window runs on the wall clock.
            now = frame.stream_time
            self._stream_time = now
            context = SceneContext.build(
                self.config.camera_id,
                datetime.fromtimestamp(frame.timestamp),
                night_start=self.config.night_start,
                night_end=self.config.night_end,
                force_night=self.config.force_night,
            )

            self.clips.observe(frame.image, now)
            self.anpr.tick(now)
            self.passes.flush_departed(now)
            self.identities.tick()
            detections = list(self.detector.detect(frame.image))
            if self.tracker is not None:
                # The frame is needed for appearance-based re-identification.
                detections = self.tracker.update(detections, frame.image)
            self.stats.detections += len(detections)

            overlays: list[tuple[Detection, Zone | None, bool]] = []
            pending: list[
                tuple[
                    Detection, Zone | None, TrackContext | None,
                    RiskAssessment, list[Behaviour], str,
                ]
            ] = []
            for detection in detections:
                zone = self._locate(detection, width, height)
                track = self._observe_context(detection, zone, context, width, height, now)

                # Risk is scored continuously for anything inside an armed
                # zone; whether that becomes an alert is a separate decision,
                # which is what lets a worsening situation re-notify.
                # The Behaviour Engine runs for every tracked person, not just
                # those inside a zone - approach and erratic movement are
                # signals precisely while someone is still outside.
                self.anpr.observe(frame.image, detection)
                # Every vehicle, in any zone, alarm or not. The pass log answers
                # "what came through here today", which alerts never do.
                if detection.track_id is not None:
                    self.passes.observe(
                        self.config.camera_id, detection, now, frame.timestamp,
                        crop=self.anpr.crop_for(detection.track_id),
                        reading=self.anpr.best_for(detection.track_id),
                    )
                behaviours = (
                    self.behaviour.classify(track, detection.label) if track else []
                )
                behaviours += self.behaviour.from_registry(
                    self._verify_plate(detection)
                )
                behaviours += self.behaviour.from_identity(
                    self.identities.observe(frame.image, detection)
                )
                self._follow_across_cameras(detection, frame.image, now)

                assessment = None
                if zone is not None and zone.kind in self.config.alert_kinds:
                    assessment = self.risk.assess(detection, zone, context, track, behaviours)

                fired = self.monitor.update(
                    detection.track_id, zone, now,
                    assessment.severity if assessment else None,
                )

                # Camera tampering is scored without a zone, because someone
                # reaching for the lens is a threat wherever they stand.
                tampering = any(b.name == CAMERA_TAMPERING for b in behaviours)
                tamper_assessment = None
                if tampering:
                    tamper_assessment = self.risk.assess(
                        detection, None, context, track, behaviours
                    )
                tamper_fired = self.monitor.update_tampering(
                    detection.track_id, now, tampering,
                    tamper_assessment.severity if tamper_assessment else None,
                )

                overlays.append((detection, zone, fired or tamper_fired))
                if not self.config.quiet and self._should_log_detection(
                    detection, zone, behaviours
                ):
                    ui.emit(
                        render_detection(
                            detection, zone, self.config.camera_id, frame.index,
                            track, behaviours,
                        )
                    )
                if fired and zone is not None and assessment is not None:
                    pending.append(
                        (detection, zone, track, assessment, behaviours, EVENT_TYPE)
                    )
                if tamper_fired and tamper_assessment is not None:
                    pending.append(
                        (detection, None, track, tamper_assessment, behaviours,
                         TAMPER_EVENT_TYPE)
                    )

            annotated = None
            # Also drawn when something is watching through a sink - the
            # dashboard streams this frame, and a viewer that only sees a
            # picture when an alert fires is not a live view.
            if pending or self.config.view or self.frame_sink is not None:
                annotated = annotate(
                    frame.image,
                    list(self.zones),
                    overlays,
                    self.config.camera_id,
                    frame.index,
                    self.stats.alerts + len(pending),
                    self.stats.fps,
                )

            for detection, zone, track, assessment, behaviours, event_type in pending:
                self._raise_alert(
                    detection, zone, context, frame.index, annotated,
                    track, assessment, behaviours, event_type,
                )

            if annotated is not None:
                # A sink means somebody else is drawing this - the fleet's own
                # window, or the dashboard streaming it. Only a lone camera
                # with a window of its own draws here, because OpenCV's GUI
                # must be driven from one thread and this is not it.
                if self.frame_sink is not None:
                    self.frame_sink(self.config.camera_id, annotated)
                elif self.config.view:
                    cv2.imshow(self._window, annotated)
                    if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                        ui.warn("Live view closed by operator.")
                        break

            self._check_confirmation_window()
            self._heartbeat(frame.index)

    def reload_zones(self) -> int:
        """Re-read the fence file while the camera keeps running.

        Drawing a fence and then restarting the post to use it is the kind of
        friction that means fences never get redrawn. The context engine holds
        a reference to the manager, so it is rebuilt against the new zones
        rather than left pointing at the old ones - which would leave the
        approach and dwell tracking judging a fence that is no longer there.
        """
        self.zones = ZoneManager.from_file(self.config.zones_path)
        self.context = ContextEngine(self.zones, armed_kinds=self.config.alert_kinds)
        ui.info(
            f"[{self.config.camera_id}] fences reloaded from "
            f"{self.config.zones_path} ({len(self.zones)} zones)"
        )
        return len(self.zones)

    def _check_confirmation_window(self) -> None:
        """Warn when the frame rate makes the confirmation window dangerous.

        `confirm_frames` exists to suppress single-frame detector flicker, so
        frames are the right unit for it. But the *time* it costs depends
        entirely on throughput: three frames is a fifth of a second on one
        camera and over two seconds when eight share a CPU - long enough for
        someone to cross a zone between the frames that would have confirmed
        them. Better to say so than to let the policy quietly stop working.
        """
        if self._confirmation_warned or self.stats.frames < 40:
            return
        fps = self.stats.fps
        if fps <= 0:
            return
        window = self.config.confirm_frames / fps
        if window < self.confirmation_warn_seconds:
            return
        self._confirmation_warned = True
        ui.warn(
            f"[{self.config.camera_id}] {fps:.1f} fps means {self.config.confirm_frames} "
            f"confirmation frames take {window:.1f}s - a running person can cross a zone "
            f"in that time. Lower --confirm-frames or run fewer cameras. "
            f"Do NOT raise --stride: it analyses fewer frames, which makes this window "
            f"longer, not shorter."
        )

    def _heartbeat(self, frame_index: int) -> None:
        if self.config.quiet:
            return
        now = time.time()
        if now - self._last_status < STATUS_INTERVAL:
            return
        self._last_status = now
        print(render_status(frame_index, self.stats.fps, len(self._zone_of_track),
                            self.stats.alerts))

    def _teardown(self) -> None:
        self.anpr.stop()
        self.passes.close()
        self.clips.flush()
        if self.config.view and self.frame_sink is None:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        if not self.config.print_summary:
            ui.info(
                f"[{self.config.camera_id}] stopped - {self.stats.frames} frames, "
                f"{self.stats.alerts} alerts"
            )
            return
        ui.section("SESSION SUMMARY")
        print(ui.field("Camera", self.config.camera_id))
        print(ui.field("Frames", str(self.stats.frames)))
        print(ui.field("Detections", str(self.stats.detections)))
        print(ui.field("Alerts", str(self.stats.alerts)))
        print(ui.field("Avg FPS", f"{self.stats.fps:.1f}"))
        if self.passes.enabled and (self.passes.written or self.passes.suppressed):
            read = "" if not self.passes.written else ""
            print(ui.field(
                "Vehicles logged",
                f"{self.passes.written} pass(es)"
                + (f", {self.passes.suppressed} duplicate(s) suppressed"
                   if self.passes.suppressed else "")
            ))
        if self.tracker is not None and self.tracker.reids:
            print(ui.field("Re-identified", f"{self.tracker.reids} track(s) recovered"))
        if self.config.anpr and self.anpr.dropped:
            print(
                ui.field(
                    "ANPR backlog", f"{self.anpr.dropped} frame(s) dropped while reading"
                )
            )
        if self._source_health is not None:
            health = self._source_health
            print(ui.field("Link state", health.state.value))
            print(
                ui.field(
                    "Link stats", f"{health.connects} connects, {health.disconnects} drops"
                )
            )
            if health.last_error:
                print(ui.field("Last error", health.last_error))
        print(ui.field("Event log", str(self.event_log.path_for(datetime.now()))))
        if self.store is not None:
            print(ui.field("Event store", f"{self.store.path} ({self.store.count()} total)"))
        if self.config.save_evidence:
            print(ui.field("Evidence", str(self.config.evidence_dir)))
        if self.config.save_clips and self.clips.written:
            print(ui.field("Clips", f"{self.clips.written} written"))
        print()
        ui.ok("Surveillance stopped.")
        print()


def run(config: SurveillanceConfig) -> SessionStats:
    ui.init(use_color=config.color)
    store = EventStore(config.db_path) if config.db_path else None
    try:
        return SurveillancePipeline(config, store=store).run()
    finally:
        if store is not None:
            store.close()
