"""Fetch the number-plate detection model.

SENTINEL-X works without it — the ANPR stage falls back to handing whole vehicles to
the text detector — but plate *localisation* is dramatically better with it. On
the same Delhi traffic footage:

    contour heuristics       43 candidates,  0 were plates
    + character counting      7 candidates,  0 were plates
    this model               30 detections, plates

    python tools/fetch_plate_model.py

The weights are a third-party model published on Hugging Face. **Check its
licence before deploying anything to a government post** — this script only
downloads it, it makes no claim about the terms it is offered under.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/"
    "resolve/main/license-plate-finetune-v1n.pt"
)
DEST = Path("models/plate-yolov11n.pt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(DEST))
    parser.add_argument("--force", action="store_true", help="download even if present")
    args = parser.parse_args(argv)

    dest = Path(args.output)
    if dest.exists() and not args.force:
        print(f"{dest} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {URL}")
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=180) as response:
            dest.write_bytes(response.read())
    except (urllib.error.URLError, OSError) as exc:
        print(f"could not download the plate model: {exc}", file=sys.stderr)
        print("SENTINEL-X will still run; plate localisation falls back to the", file=sys.stderr)
        print("text detector, which finds plates far less reliably.", file=sys.stderr)
        return 1

    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    try:
        from ultralytics import YOLO

        print("classes:", YOLO(str(dest)).names)
    except Exception as exc:
        print(f"downloaded, but the model would not load: {exc}", file=sys.stderr)
        return 1

    print()
    print("Check the model's licence before deploying it operationally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
