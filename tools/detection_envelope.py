"""Measure how small a person can get before detection fails.

This answers the question a border post actually has to answer before mounting
anything: *how far down the approach will this camera still see someone?*

The model does not care about metres, it cares about pixels. So the sweep
scales a person to a range of apparent heights and reports where detection
stops, by day and under a night simulation. Convert that to distance with the
camera's own field of view:

    person height in pixels  =  frame height x 1.7 m / (2 x range x tan(vFOV/2))

    python tools/detection_envelope.py
    python tools/detection_envelope.py --sweep      # speed against range
    python tools/detection_envelope.py --weights models/yolov8s.pt

Numbers move a little between runs - the night grain is random and detection
near a floor is marginal by definition. Raise --repeats for steadier figures.

Two honest limits on what this produces. The night column is a *simulation* -
monochrome, sensor grain and motion blur applied to daylight footage - and real
infrared differs in ways that cannot be faked here, because IR reflectance is
not visible reflectance. And it samples one person against one background, so
treat the numbers as an order of magnitude, not a specification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SAMPLE = Path("data/samples/bus.jpg")
PERSON_CROP = (40, 370, 250, 920)      # x1, y1, x2, y2 - one full-body figure
FRAME_W, FRAME_H = 960, 540
HEIGHTS = (400, 300, 220, 160, 120, 90, 70, 55, 40, 30)


def night(frame: np.ndarray, grain: float = 20.0, blur: int = 7) -> np.ndarray:
    """Monochrome, grainy and motion-blurred: the parts of night we can fake."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out += np.random.normal(0, grain, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    kernel = np.zeros((blur, blur), np.float32)
    kernel[blur // 2, :] = 1.0 / blur
    return cv2.filter2D(out, -1, kernel)


def compose(background: np.ndarray, person: np.ndarray, height: int) -> np.ndarray:
    ph, pw = person.shape[:2]
    scale = height / ph
    scaled = cv2.resize(
        person, (max(1, int(pw * scale)), height), interpolation=cv2.INTER_AREA
    )
    h, w = scaled.shape[:2]
    frame = background.copy()
    y, x = FRAME_H - h - 20, (FRAME_W - w) // 2
    frame[y:y + h, x:x + w] = scaled
    return frame


SWEEP_SIZES = (960, 736, 640, 512, 416, 320)


def sweep(model, person, background, args) -> int:
    """Speed against range, so the trade can be made with numbers.

    Smaller input is faster and sees less far. There is no single right answer:
    it depends on how far down the approach the camera has to work.
    """
    import time

    print(f"model: {args.weights}   conf>={args.confidence}   "
          f"{args.repeats} runs per height")
    print()
    print(f"{'imgsz':>6}{'ms/frame':>10}{'day floor':>11}{'night floor':>13}")

    blank = background.copy()
    for size in SWEEP_SIZES:
        model.predict(blank, classes=[0], imgsz=size, verbose=False)
        start = time.perf_counter()
        for _ in range(6):
            model.predict(blank, classes=[0], imgsz=size, verbose=False)
        per_frame = (time.perf_counter() - start) / 6 * 1000

        day_floor = night_floor = None
        for height in HEIGHTS:
            frame = compose(background, person, height)
            runs = max(1, args.repeats)

            def hits(builder) -> int:
                found = 0
                for _ in range(runs):
                    result = model.predict(
                        builder(), classes=[0], conf=args.confidence,
                        imgsz=size, verbose=False,
                    )[0]
                    if result.boxes is not None and len(result.boxes):
                        found += 1
                return found

            if hits(lambda: frame) == runs:
                day_floor = height
            if hits(lambda: night(frame)) == runs:
                night_floor = height

        print(f"{size:>6}{per_frame:>10.0f}{str(day_floor or '-'):>11}"
              f"{str(night_floor or '-'):>13}")

    print()
    print("Pick the largest input you can afford that still reaches the person")
    print("furthest from the camera - at the NIGHT floor, not the day one.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default="models/yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=512,
                        help="input size to test; matches the pipeline default")
    parser.add_argument("--repeats", type=int, default=3,
                        help="runs per height; night grain is random")
    parser.add_argument("--sweep", action="store_true",
                        help="compare input sizes: speed against how small a person can be")
    args = parser.parse_args(argv)

    if not SAMPLE.exists():
        print(f"missing {SAMPLE}; run tools/make_demo_video.py first", file=sys.stderr)
        return 1
    image = cv2.imread(str(SAMPLE))
    if image is None:
        print(f"could not decode {SAMPLE}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    model = YOLO(args.weights)
    x1, y1, x2, y2 = PERSON_CROP
    person = image[y1:y2, x1:x2].copy()
    background = cv2.GaussianBlur(
        cv2.resize(image[880:1080, 260:810], (FRAME_W, FRAME_H)), (0, 0), 4
    )

    def hit_rate(frame_builder) -> tuple[float, float]:
        hits, confidences = 0, []
        for _ in range(max(1, args.repeats)):
            result = model.predict(
                frame_builder(), classes=[0], conf=args.confidence,
                imgsz=args.imgsz, verbose=False,
            )[0]
            boxes = result.boxes
            if boxes is not None and len(boxes):
                hits += 1
                confidences.append(float(boxes.conf.cpu().numpy().max()))
        runs = max(1, args.repeats)
        return hits / runs, (sum(confidences) / len(confidences) if confidences else 0.0)

    if args.sweep:
        return sweep(model, person, background, args)

    print(f"model: {args.weights}   conf>={args.confidence}   imgsz={args.imgsz}")
    print()
    print(f"{'person px':>10}{'day':>8}{'conf':>7}{'night':>8}{'conf':>7}")
    day_floor = night_floor = None
    for height in HEIGHTS:
        frame = compose(background, person, height)
        day_rate, day_conf = hit_rate(lambda: frame)
        night_rate, night_conf = hit_rate(lambda: night(frame))

        if day_rate >= 0.99:
            day_floor = height
        if night_rate >= 0.99:
            night_floor = height
        print(f"{height:>10}{day_rate:>8.0%}{day_conf:>7.2f}{night_rate:>8.0%}{night_conf:>7.2f}")

    print()
    print(f"reliable down to : {day_floor or '-'} px by day, "
          f"{night_floor or '-'} px under the night simulation")
    print("Night costs pixels. Size a lens so the furthest person you care about")
    print("still fills the night figure, not the day one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
