"""Fetch the face detection and recognition models.

Both come from the OpenCV model zoo and are used through OpenCV's own APIs, so
they add no Python dependency - only two files.

    python tools/fetch_face_models.py

Face recognition is a *checkpoint* capability, not a perimeter one: a face
needs roughly 60-80 pixels across before the vector means anything, and a
person 200 px tall has a face of about 25. Run tools/face_envelope.py to see
where the limit falls for your camera before relying on it.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Running a file inside tools/ puts tools/ on the path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    Path("models/face_detection_yunet.onnx"):
        f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    Path("models/face_recognition_sface.onnx"):
        f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download even if present")
    args = parser.parse_args(argv)

    for dest, url in MODELS.items():
        if dest.exists() and not args.force:
            print(f"{dest} already present ({dest.stat().st_size / 1e6:.2f} MB)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching {dest.name}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=180) as response:
                dest.write_bytes(response.read())
        except (urllib.error.URLError, OSError) as exc:
            print(f"could not download {dest.name}: {exc}", file=sys.stderr)
            print("Face recognition stays off; everything else runs.", file=sys.stderr)
            return 1
        print(f"  wrote {dest} ({dest.stat().st_size / 1e6:.2f} MB)")

    try:
        from app.faces import FaceEngine

        engine = FaceEngine()
        if not engine.available:
            print(f"models downloaded but would not load: {engine.last_error}",
                  file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"models downloaded but would not load: {exc}", file=sys.stderr)
        return 1

    print()
    print("Enrol the people this post expects to see:")
    print("  python tools/enrol_face.py --id SSB-114 --label 'Havildar Singh' photo.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
