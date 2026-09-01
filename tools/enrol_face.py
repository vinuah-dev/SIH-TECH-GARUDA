"""Enrol somebody the post expects to see.

    python tools/enrol_face.py --id SSB-114 --label "Havildar Singh" \\
        --role "day patrol" photo1.jpg photo2.jpg

Several photographs from different angles make recognition far steadier than
one, because a face vector is sensitive to pose. Two or three is usually enough.

What ends up in the roster file is a list of numbers per photograph. The
photographs are read and discarded; nothing that can be turned back into an
image of somebody's face is written anywhere.

Enrolment is deliberately a separate, manual step. Nothing in the running
pipeline ever adds a person it merely saw - a system that quietly builds a
biometric database of passers-by is a different system than this one, and it is
not one this project will become by accident.

Who may lawfully be enrolled, and on what basis, is a question for the
deploying authority, not for this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.faces import FaceEngine  # noqa: E402
from app.faces.roster import FaceRoster, Person  # noqa: E402

DEFAULT_ROSTER = Path("config/face-roster.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos", nargs="+", help="one or more photographs of one person")
    parser.add_argument("--id", dest="person_id", required=True,
                        help="identifier recorded in events, e.g. SSB-114")
    parser.add_argument("--label", default=None, help="name shown to the operator")
    parser.add_argument("--role", default="", help="e.g. 'day patrol'")
    parser.add_argument("--note", default="")
    parser.add_argument("--roster", default=str(DEFAULT_ROSTER))
    parser.add_argument("--replace", action="store_true",
                        help="overwrite this person instead of refusing")
    args = parser.parse_args(argv)

    engine = FaceEngine()
    if not engine.available:
        print(f"face models unavailable: {engine.last_error}", file=sys.stderr)
        print("Run tools/fetch_face_models.py first.", file=sys.stderr)
        return 1

    roster_path = Path(args.roster)
    roster = FaceRoster.from_file(roster_path) if roster_path.exists() else FaceRoster()

    existing = next((p for p in roster if p.person_id == args.person_id), None)
    if existing and not args.replace:
        print(f"{args.person_id} is already enrolled as {existing.label!r}. "
              f"Pass --replace to overwrite.", file=sys.stderr)
        return 1

    vectors = []
    for photo in args.photos:
        image = cv2.imread(photo)
        if image is None:
            print(f"could not read {photo}", file=sys.stderr)
            return 1

        faces = engine.detect(image)
        if not faces:
            print(f"{photo}: no face found", file=sys.stderr)
            continue
        if len(faces) > 1:
            print(f"{photo}: {len(faces)} faces found; using the largest "
                  f"({faces[0].width} px wide)")

        face = faces[0]
        if face.width < 80:
            print(f"{photo}: face is only {face.width} px wide - too small to enrol "
                  f"reliably, use a closer photograph", file=sys.stderr)
            continue

        vector = engine.embed(image, face)
        if vector is None:
            print(f"{photo}: could not build a face vector", file=sys.stderr)
            continue
        vectors.append(vector)
        print(f"{photo}: enrolled from a {face.width} px face")

    if not vectors:
        print("nothing enrolled", file=sys.stderr)
        return 1

    roster.add(
        Person(
            person_id=args.person_id,
            label=args.label or args.person_id,
            vectors=tuple(vectors),
            role=args.role,
            note=args.note,
        )
    )
    written = roster.save(roster_path)
    print()
    print(f"{args.person_id} enrolled from {len(vectors)} photograph(s) -> {written}")
    print(f"roster now holds {len(roster)} person(s)")
    print()
    print("Run with:  python main.py --source ... --faces", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
