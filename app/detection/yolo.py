"""YOLOv8 object detector (Ultralytics).

Only the classes IBVAP is configured to watch are requested from the model:
filtering at inference time is cheaper than discarding results afterwards, and
it keeps unrelated COCO classes out of the tracker entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from . import classes
from .base import Detection


class YoloObjectDetector:
    """Wraps an Ultralytics YOLO model behind the `Detector` protocol."""

    name = "YOLOv8n"
    # Plain prediction is stateless, so one model can serve every camera.
    # ByteTrack is not: it keeps track state per stream, and sharing it would
    # let one camera inherit another camera's track ids.
    shareable = True

    def __init__(
        self,
        weights: str | Path = "models/yolov8n.pt",
        confidence: float = 0.40,
        imgsz: int = 640,
        device: str = "cpu",
        use_bytetrack: bool = False,
        labels: Sequence[str] | None = None,
    ) -> None:
        # Imported lazily so the zone/risk/alert unit tests never need torch.
        from ultralytics import YOLO

        self.weights = str(weights)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.use_bytetrack = use_bytetrack
        self.labels = list(labels) if labels else list(classes.CATEGORIES[classes.PERSON])
        self._class_ids = classes.coco_ids(self.labels)
        if not self._class_ids:
            raise ValueError("no known object classes selected for detection")

        self.model = YOLO(self.weights)
        watching = ", ".join(self.labels)
        self.name = f"YOLOv8 ({Path(self.weights).name}) watching {watching}"
        self.shareable = not use_bytetrack

    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        kwargs = dict(
            source=frame,
            classes=self._class_ids,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if self.use_bytetrack:
            results = self.model.track(persist=True, tracker="bytetrack.yaml", **kwargs)
        else:
            results = self.model.predict(**kwargs)

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy() if getattr(boxes, "id", None) is not None else None
            for i in range(len(xyxy)):
                entry = classes.BY_ID.get(int(class_ids[i]))
                if entry is None:
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in xyxy[i])
                detections.append(
                    Detection(
                        label=entry.label,
                        confidence=float(confs[i]),
                        bbox=(x1, y1, x2, y2),
                        track_id=int(ids[i]) if ids is not None else None,
                    )
                )
        return detections


# The detector was person-only for its first few stages; keep the old name
# working so nothing outside this package has to care.
YoloPersonDetector = YoloObjectDetector
