"""Build a small demo clip for the virtual-fence walkthrough.

Composites a real person crop (from data/samples/bus.jpg) walking left to
right across a pavement background, so `main.py --source <clip>` exercises
genuine YOLO inference rather than the simulator.

    python tools/make_demo_video.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

SOURCE = Path("data/samples/bus.jpg")
SOURCE_URL = "https://ultralytics.com/images/bus.jpg"
OUTPUT = Path("data/samples/border_demo.mp4")
WIDTH, HEIGHT, FPS, FRAMES = 960, 540, 12, 72

# Region of bus.jpg containing a single full-body pedestrian.
PERSON_CROP = (40, 370, 250, 920)  # x1, y1, x2, y2


def build_background(image: np.ndarray) -> np.ndarray:
    """Blurred pavement strip, used as a neutral moving-target backdrop."""
    pavement = image[880:1080, 260:810]
    background = cv2.resize(pavement, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(background, (0, 0), 3)


def fetch_source() -> bool:
    """Pull the sample pedestrian image if it is not already on disk.

    The image is generated output, not committed, so a fresh checkout has to
    fetch it once before the clip can be built.
    """
    if SOURCE.exists():
        return True
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {SOURCE_URL} -> {SOURCE}")
    try:
        urllib.request.urlretrieve(SOURCE_URL, SOURCE)
    except (urllib.error.URLError, OSError) as exc:
        print(f"could not download the sample image: {exc}", file=sys.stderr)
        print(
            f"Put any photo containing a full-body person at {SOURCE} and re-run, "
            "adjusting PERSON_CROP to frame them.",
            file=sys.stderr,
        )
        return False
    return True


def main() -> int:
    if not fetch_source():
        return 1

    image = cv2.imread(str(SOURCE))
    if image is None:
        print(f"could not decode {SOURCE}", file=sys.stderr)
        return 1

    background = build_background(image)
    x1, y1, x2, y2 = PERSON_CROP
    person = image[y1:y2, x1:x2]

    target_h = int(HEIGHT * 0.62)
    scale = target_h / person.shape[0]
    person = cv2.resize(person, (int(person.shape[1] * scale), target_h),
                        interpolation=cv2.INTER_AREA)
    ph, pw = person.shape[:2]
    feet_y = int(HEIGHT * 0.92)
    top = feet_y - ph

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        print("could not open the video writer", file=sys.stderr)
        return 1

    start_x, end_x = int(WIDTH * 0.04), int(WIDTH * 0.80)
    for i in range(FRAMES):
        progress = i / max(1, FRAMES - 1)
        left = int(start_x + (end_x - start_x) * progress)
        frame = background.copy()
        right = min(WIDTH, left + pw)
        frame[top:top + ph, left:right] = person[:, : right - left]
        writer.write(frame)

    writer.release()
    print(f"wrote {OUTPUT} ({FRAMES} frames @ {FPS} fps, {WIDTH}x{HEIGHT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
