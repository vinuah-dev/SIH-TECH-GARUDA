"""Measure how large a face has to be before recognition means anything.

Face recognition degrades differently from detection. Detection fails visibly -
no box. Recognition fails *quietly*: it still returns a vector, and that vector
still produces a similarity score, and the score can clear a threshold by luck.
A system that recognises the wrong guard is worse than one that recognises
nobody, so the number that matters is the width below which matching stops
being trustworthy.

The test is deliberately generous, and that is the point of it. It matches a
face against *itself*, scaled down - the same pixels, the same pose, the same
light - so the errors on both sides are correlated and the score is far kinder
than reality. Read the result as a **ceiling on optimism**, never as a working
threshold:

    fails to match itself at N px  ->  will certainly fail at N px in the field
    matches itself at N px         ->  says nothing about a real second photo

Real recognition compares today's face to one enrolled weeks ago, at a
different angle under different light. That is a much harder problem than this
tool measures, and the usable width is well above whatever floor it reports.

    python tools/face_envelope.py --photo somebody.jpg

With no photograph it uses the bundled sample, whose faces are small - which is
itself the point being demonstrated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.faces import FaceEngine  # noqa: E402
from app.faces.roster import DEFAULT_THRESHOLD  # noqa: E402

SAMPLE = Path("data/samples/bus.jpg")
WIDTHS = (200, 160, 120, 100, 80, 70, 60, 50, 40, 30, 20)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photo", default=str(SAMPLE))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    engine = FaceEngine()
    if not engine.available:
        print(f"face models unavailable: {engine.last_error}", file=sys.stderr)
        print("Run tools/fetch_face_models.py first.", file=sys.stderr)
        return 1

    image = cv2.imread(args.photo)
    if image is None:
        print(f"could not read {args.photo}", file=sys.stderr)
        return 1

    engine.min_width = 12          # detect everything, judge it here instead
    faces = engine.detect(image)
    if not faces:
        print(f"no face found in {args.photo}", file=sys.stderr)
        return 1

    face = faces[0]
    x1, y1, x2, y2 = face.bbox
    # A generous margin, because alignment wants the whole head.
    pad = int(face.width * 0.6)
    h, w = image.shape[:2]
    crop = image[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]

    reference = engine.embed(image, face)
    if reference is None:
        print("could not build a reference vector", file=sys.stderr)
        return 1

    print(f"photo     : {args.photo}")
    print(f"reference : face {face.width} px wide")
    print(f"threshold : {args.threshold} (cosine similarity)")
    print()
    if face.width < 100:
        print(f"NOTE: the reference face is only {face.width} px, so this sweep can only")
        print("      test a few small sizes. Use a close portrait for a meaningful curve.")
        print()
    print(f"{'face px':>8}{'detected':>10}{'similarity':>12}{'matches':>9}")

    floor = None
    for width in WIDTHS:
        if width > face.width:
            continue
        scale = width / face.width
        small = cv2.resize(
            crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        found = engine.detect(small)
        if not found:
            print(f"{width:>8}{'no':>10}{'-':>12}{'-':>9}")
            continue

        vector = engine.embed(small, found[0])
        score = engine.similarity(reference, vector)
        matches = score >= args.threshold
        if matches:
            floor = width
        print(f"{width:>8}{'yes':>10}{score:>12.3f}{('yes' if matches else 'NO'):>9}")

    print()
    if floor:
        print(f"Matched itself down to {floor} px - a BEST CASE, not a working size.")
        print("The same pixels are on both sides of that comparison. Against a")
        print("photograph enrolled on another day the usable width is well above it,")
        print("which is why the engine refuses faces under "
              f"{FaceEngine.__dataclass_fields__['min_width'].default} px by default.")
    else:
        print("Never matched itself at any tested size - the reference face is too small")
        print("to enrol from at all.")
    print()
    print("A face is roughly one eighth of a person's height, so a face of N px")
    print("needs a person of about 8N px in frame. Perimeter cameras do not")
    print("deliver that; a gate or checkpoint camera does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
