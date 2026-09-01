"""A detector trained to find number plates, and nothing else.

Three approaches to locating a plate were tried against the same Delhi traffic
footage, and the difference is not marginal:

    contour heuristics       43 candidates,  0 were plates
    + character counting      7 candidates,  0 were plates
    plate detection model    30 detections, plates

Shape and texture cannot separate a plate from the grille slats, bumper
shadows and badge recesses around it. A model trained on plates can, and it
costs 5.5 MB and one small forward pass.

The model is optional. Without it the engine falls back to handing the whole
vehicle to the text detector, which is worse but still works. Nothing here
downloads anything at runtime - see `tools/fetch_plate_model.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

DEFAULT_WEIGHTS = Path("models/plate-yolov11n.pt")


@dataclass(frozen=True)
class PlateBox:
    """One detected plate, in full-frame coordinates."""

    bbox: tuple[int, int, int, int]
    confidence: float

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class PlateDetector:
    """Locates number plates with a purpose-trained YOLO model."""

    weights: Path = DEFAULT_WEIGHTS
    confidence: float = 0.30
    imgsz: int = 960
    device: str = "cpu"
    # A plate narrower than this has too few pixels per character for any
    # recogniser to do better than guess.
    min_width: int = 60
    # When the whole-frame pass finds no plate on a vehicle, look again at the
    # vehicle alone, enlarged. Measured on 308 vehicles of real 848x478 phone
    # footage: the frame pass alone found 15 plates (4.9%); adding this found
    # 135 (43.8%). Nine times more, because at that resolution a plate is about
    # 40 px in the full frame and the model simply cannot see it there.
    #
    # This does not contradict the note on `detect_on` - an unenlarged crop
    # does cost recall. What works is cropping *and* magnifying, which puts
    # more pixels on the plate than the frame pass ever had.
    rescue_crops: bool = True
    rescue_scale: float = 3.0
    # The rescue pass is looking at an enlarged crop, so the floor that applies
    # to the frame pass would throw away everything it is there to find.
    rescue_min_width: int = 12
    # Geometry the rescue pass has to satisfy, because it does not get one for
    # free from the frame pass.
    #
    # Handed a magnified, blurred vehicle crop - which is nothing like the
    # scenes it was trained on - the model returns boxes covering most of the
    # crop. Inspected, those were a video watermark across a bus and a TATA
    # badge, not plates. The first version of this rescue counted them as
    # successes and reported twice the recall; they were false positives.
    #
    # A plate is a small part of the vehicle carrying it, and it is much wider
    # than it is tall. Both are true regardless of camera, so both are checked.
    rescue_max_vehicle_fraction: float = 0.5
    rescue_min_aspect: float = 1.6     # a two-line plate is roughly 2:1
    rescue_max_aspect: float = 7.0     # a single-line Indian plate about 4.7:1

    model: object | None = field(default=None, init=False)
    available: bool = field(default=False, init=False)
    last_error: str | None = field(default=None, init=False)
    # Plates found in the frame currently being processed. Detection is a full
    # forward pass, and every vehicle in the frame wants the same answer, so
    # running it per vehicle was paying for the same work three or four times.
    _cache_key: object = field(default=None, init=False)
    _cache: list = field(default_factory=list, init=False)
    passes: int = field(default=0, init=False)
    rescues: int = field(default=0, init=False)
    rescued: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.weights = Path(self.weights)
        if not self.weights.exists():
            self.last_error = f"{self.weights} not present"
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.weights))
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on the install
            self.last_error = str(exc)

    def detect(self, frame: np.ndarray, cache_key: object = None) -> Sequence[PlateBox]:
        """Every plate in the frame, most confident first.

        `cache_key` identifies the frame. Pass the same key for every vehicle
        in one frame and the model runs once, not once per vehicle.
        """
        if not self.available or frame is None or frame.size == 0:
            return []
        if cache_key is not None and cache_key == self._cache_key:
            return self._cache
        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            self.last_error = str(exc)
            return []

        found: list[PlateBox] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = (int(round(v)) for v in xyxy[i])
                if x2 - x1 < self.min_width:
                    continue
                found.append(PlateBox(bbox=(x1, y1, x2, y2), confidence=float(confs[i])))

        found.sort(key=lambda p: -p.confidence)
        self.passes += 1
        if cache_key is not None:
            self._cache_key, self._cache = cache_key, found
        return found

    def detect_on(
        self,
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int],
        cache_key: object = None,
    ) -> Sequence[PlateBox]:
        """Plates belonging to one vehicle.

        Detection runs on the whole frame and results are then attributed to
        the vehicle, rather than running on a crop: the model was trained on
        whole scenes, and a tight crop measurably costs it recall.
        """
        vx1, vy1, vx2, vy2 = vehicle_bbox
        inside = []
        for plate in self.detect(frame, cache_key=cache_key):
            px1, py1, px2, py2 = plate.bbox
            cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
            if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                inside.append(plate)
        if inside or not self.rescue_crops:
            return inside
        return self._rescue(frame, vehicle_bbox)

    def _rescue(
        self, frame: np.ndarray, vehicle_bbox: tuple[int, int, int, int]
    ) -> Sequence[PlateBox]:
        """Look again at one vehicle, enlarged, when the frame pass found nothing.

        Costs a second forward pass for that vehicle, so it runs only where the
        cheap pass already failed. On footage where the frame pass works this
        is almost never reached; on low-resolution footage it is what makes
        plates findable at all.
        """
        if not self.available:
            return []
        vx1, vy1, vx2, vy2 = vehicle_bbox
        h, w = frame.shape[:2]
        vx1, vy1 = max(0, vx1), max(0, vy1)
        vx2, vy2 = min(w, vx2), min(h, vy2)
        patch = frame[vy1:vy2, vx1:vx2]
        if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
            return []

        scale = self.rescue_scale
        enlarged = cv2.resize(patch, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        try:
            results = self.model.predict(
                source=enlarged, conf=self.confidence, imgsz=self.imgsz,
                device=self.device, verbose=False,
            )
        except Exception as exc:
            self.last_error = str(exc)
            return []

        found: list[PlateBox] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                # Back to frame coordinates: undo the magnification, then undo
                # the crop. Getting this wrong would file a plate crop taken
                # from the wrong part of the picture.
                x1, y1, x2, y2 = (v / scale for v in xyxy[i])
                box = (int(round(vx1 + x1)), int(round(vy1 + y1)),
                       int(round(vx1 + x2)), int(round(vy1 + y2)))
                if not self._plate_shaped(box, vehicle_bbox):
                    continue
                found.append(PlateBox(bbox=box, confidence=float(confs[i])))

        found.sort(key=lambda p: -p.confidence)
        self.rescues += 1
        self.rescued += len(found)
        return found

    def _plate_shaped(
        self, box: tuple[int, int, int, int], vehicle: tuple[int, int, int, int]
    ) -> bool:
        """Could this box be a plate on this vehicle, by geometry alone?

        Nothing here looks at pixels. It rejects the shapes a plate cannot
        have: too narrow to read, taller than it is wide, or so large that it
        would be most of the vehicle. That last one is what a watermark across
        a bus looks like.
        """
        width, height = box[2] - box[0], box[3] - box[1]
        if width < self.rescue_min_width or height <= 0:
            return False

        aspect = width / height
        if not self.rescue_min_aspect <= aspect <= self.rescue_max_aspect:
            return False

        vehicle_width = vehicle[2] - vehicle[0]
        if vehicle_width > 0 and width > vehicle_width * self.rescue_max_vehicle_fraction:
            return False
        return True
