"""Runtime configuration for one surveillance session."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SurveillanceConfig:
    # Camera / input
    camera_id: str = "CAM-01"
    source: str = "synthetic"
    stride: int = 1
    limit: int | None = None
    loop: bool = False
    max_width: int | None = 960
    synthetic_frames: int = 60
    reconnect_attempts: int = 0  # 0 = retry a dropped stream forever
    sim_walk_fraction: float = 1.0
    sim_approach_camera: float = 0.0

    # Detection
    detector: str = "yolo"  # yolo | sim
    weights: str = "models/yolov8n.pt"
    confidence: float = 0.40
    detect: tuple[str, ...] = ("person",)   # person | vehicle | all | a single label
    anpr: bool = True          # read number plates off detected vehicles
    anpr_background: bool = False  # only helps when detection runs on a GPU
    # Log every vehicle that passes, not only the ones that alarm. A post
    # needs "what came through today", which alerts alone never answer.
    plate_log: bool = True
    # How long the same vehicle stays suppressed. Past it, the same plate
    # is a genuine second crossing rather than a duplicate frame.
    plate_log_window: float = 10.0
    plate_weights: Path = Path("models/plate-yolov11n.pt")   # optional; see tools/
    # When the whole-frame plate pass finds nothing on a vehicle, look again at
    # that vehicle alone, enlarged. Measured on the Delhi clip: plate evidence
    # went from 2 vehicles in 71 to 15 in 66. It costs a second model pass for
    # each vehicle the cheap pass missed, so it can be turned off.
    plate_rescue: bool = True
    # Which recogniser reads the plate crop. None picks the best that loads:
    # PaddleOCR's recogniser alone reads 73.2% of real Indian plates exactly
    # against EasyOCR's 34.5%, and at 0.22s a plate against 0.56s - see
    # app/anpr/readers.py for the measurement and why it is both.
    ocr_backend: str | None = None
    registry_path: Path | None = None   # vehicle registration extract to verify against
    faces_path: Path | None = None      # roster of people this post expects to see
    # Measured, not guessed: 512 runs 32% faster than 640 (73 ms vs 108 ms a
    # frame) and detects a person down to exactly the same size - 120 px by
    # day, 220 px under the night simulation. Going below costs range: 320 is
    # faster again but loses everything under 220 px in daylight.
    # `python tools/detection_envelope.py --sweep` re-measures this.
    imgsz: int = 512
    device: str = "cpu"
    tracker: str = "iou"  # iou | bytetrack
    reid: bool = True     # recover a track that was occluded

    # Zones
    zones_path: Path = Path("config/zones.json")
    alert_kinds: frozenset[str] = frozenset({"RESTRICTED"})

    # Alerting
    confirm_frames: int = 3
    cooldown_seconds: float = 30.0

    # Context
    night_start: int = 18
    night_end: int = 6
    force_night: bool | None = None

    # Output
    view: bool = False
    verbose: bool = False
    quiet: bool = False       # suppress per-frame console output (server mode)
    print_summary: bool = True  # off for fleet members; the fleet prints one
    save_evidence: bool = True
    save_clips: bool = True       # a few seconds of footage either side of an alert
    clip_pre_seconds: float = 4.0
    clip_post_seconds: float = 4.0
    evidence_dir: Path = field(default_factory=lambda: Path("data/evidence"))
    events_dir: Path = field(default_factory=lambda: Path("data/events"))
    db_path: Path | None = None
    color: bool = True

    def __post_init__(self) -> None:
        self.zones_path = Path(self.zones_path)
        self.evidence_dir = Path(self.evidence_dir)
        self.events_dir = Path(self.events_dir)
        if self.db_path is not None:
            self.db_path = Path(self.db_path)
        if self.registry_path is not None:
            self.registry_path = Path(self.registry_path)
        if self.faces_path is not None:
            self.faces_path = Path(self.faces_path)
        self.plate_weights = Path(self.plate_weights)
        from .detection.classes import resolve

        self.labels = resolve(list(self.detect))
        if self.detector not in {"yolo", "sim"}:
            raise ValueError("detector must be 'yolo' or 'sim'")
        if self.tracker not in {"iou", "bytetrack"}:
            raise ValueError("tracker must be 'iou' or 'bytetrack'")
        if self.confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")
