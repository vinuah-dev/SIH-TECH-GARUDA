"""Build the vehicle / ANPR / cloned-plate demo clip.

Composites a real vehicle (from the sample photo) carrying a plate that the
bundled registry extract says belongs to something else entirely, and drives it
across a virtual fence. That is the whole cloned-plate story in one 6-second
clip, and it is the demonstration worth showing:

    python tools/make_vehicle_demo.py
    python main.py --source data/samples/vehicle_demo.mp4 --detect all \\
        --registry config/vehicle-registry.json --force-night --db

The plate is drawn on rather than photographed, and it is drawn crisply. That
is deliberate and worth saying out loud: it makes the *pipeline* demonstrable
end to end, but it does not demonstrate that OCR works on real plates. It does
not - see the ANPR section of the README.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

SOURCE = Path("data/samples/bus.jpg")
SOURCE_URL = "https://ultralytics.com/images/bus.jpg"
OUTPUT = Path("data/samples/vehicle_demo.mp4")

WIDTH, HEIGHT, FPS, FRAMES = 960, 540, 10, 60

# Region of the sample photo holding the vehicle.
VEHICLE_CROP = (20, 210, 800, 640)
VEHICLE_SIZE = (520, 290)

# Where the plate sits on the resized vehicle.
PLATE_BOX = (190, 232, 165, 46)     # x, y, w, h

# Registered to a MOTORCYCLE in config/vehicle-registry.json, so a bus wearing
# it is a mismatch - which is the point of the clip.
DEFAULT_PLATE = "DL8CAF5030"


def fetch_source() -> bool:
    if SOURCE.exists():
        return True
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {SOURCE_URL} -> {SOURCE}")
    try:
        urllib.request.urlretrieve(SOURCE_URL, SOURCE)
    except (urllib.error.URLError, OSError) as exc:
        print(f"could not download the sample image: {exc}", file=sys.stderr)
        print(f"Put any photo containing a vehicle at {SOURCE} and re-run.", file=sys.stderr)
        return False
    return True


def build_vehicle(image: np.ndarray, plate: str) -> np.ndarray:
    x1, y1, x2, y2 = VEHICLE_CROP
    vehicle = cv2.resize(image[y1:y2, x1:x2].copy(), VEHICLE_SIZE, interpolation=cv2.INTER_AREA)

    px, py, pw, ph = PLATE_BOX
    cv2.rectangle(vehicle, (px, py), (px + pw, py + ph), (245, 245, 245), -1)
    cv2.rectangle(vehicle, (px, py), (px + pw, py + ph), (30, 30, 30), 2)

    # Fit the text to the plate rather than trusting one hard-coded scale.
    scale, thickness = 0.62, 2
    (tw, th), _ = cv2.getTextSize(plate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    while tw > pw - 14 and scale > 0.25:
        scale -= 0.04
        (tw, th), _ = cv2.getTextSize(plate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(
        vehicle, plate,
        (px + (pw - tw) // 2, py + (ph + th) // 2),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (12, 12, 12), thickness,
    )
    return vehicle


def build_background(image: np.ndarray) -> np.ndarray:
    pavement = image[880:1080, 260:810]
    return cv2.GaussianBlur(
        cv2.resize(pavement, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR), (0, 0), 4
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plate", default=DEFAULT_PLATE, help="plate to print on the vehicle")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)

    if not fetch_source():
        return 1
    image = cv2.imread(str(SOURCE))
    if image is None:
        print(f"could not decode {SOURCE}", file=sys.stderr)
        return 1

    vehicle = build_vehicle(image, args.plate.upper())
    background = build_background(image)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        print("could not open the video writer", file=sys.stderr)
        return 1

    vh, vw = vehicle.shape[:2]
    top = HEIGHT - vh - 30
    start_x, end_x = 30, WIDTH - vw - 60
    try:
        for i in range(FRAMES):
            frame = background.copy()
            x = int(start_x + (end_x - start_x) * i / max(1, FRAMES - 1))
            frame[top:top + vh, x:x + vw] = vehicle
            writer.write(frame)
    finally:
        writer.release()

    print(f"wrote {output} ({FRAMES} frames @ {FPS} fps, {WIDTH}x{HEIGHT})")
    print(f"plate printed: {args.plate.upper()}")
    print()
    print("Run it with:")
    print(f"  python main.py --source {output} --detect all \\")
    print("      --registry config/vehicle-registry.json --force-night --db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
