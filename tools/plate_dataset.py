"""Cut a labelled plate dataset out of the annotated photographs.

The annotations give a box and the number on the plate inside it. Training a
recogniser - or measuring one - needs those as separate files: a cropped plate
image, and the string it says.

    python tools/plate_dataset.py
    python tools/plate_dataset.py --out data/datasets/crops --min-width 60

Two splits are written, and the split is by **plate number, not by photograph**.
Several photographs of one car exist in this data - one number appears 41 times
- and letting the same plate fall on both sides of the split would let a model
score well by memorising rather than reading. That is the single easiest way to
produce a training result that means nothing.

A `labels.jsonl` in each split records the crop, the number, the source
photograph and the box, so any result can be traced back to the pixels it came
from.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_ROOT = Path("data/datasets/indian-plates")
DEFAULT_OUT = Path("data/datasets/crops")


def annotations(root: Path):
    """Every (photograph, box, number) the dataset carries."""
    for xml in sorted(root.rglob("*.xml")):
        try:
            tree = ET.parse(xml)
        except ET.ParseError:
            continue

        name = tree.findtext("filename") or ""
        image = None
        for candidate in (xml.parent / name, xml.with_suffix(".jpg"),
                          xml.with_suffix(".png"), xml.with_suffix(".jpeg")):
            if candidate.exists():
                image = candidate
                break
        if image is None:
            continue

        for obj in tree.findall("object"):
            text = "".join(
                ch for ch in (obj.findtext("name") or "").upper() if ch.isalnum()
            )
            box = obj.find("bndbox")
            if box is None or len(text) < 4 or text == "LICENCE":
                continue
            try:
                bbox = (
                    int(float(box.findtext("xmin"))), int(float(box.findtext("ymin"))),
                    int(float(box.findtext("xmax"))), int(float(box.findtext("ymax"))),
                )
            except (TypeError, ValueError):
                continue
            yield image, bbox, text


def split_of(plate: str, holdout: float) -> str:
    """Which split a plate number belongs to - stable, and by number.

    Hashing the number rather than shuffling means the same plate lands in the
    same split on every run, so two benchmark results are comparable.
    """
    digest = hashlib.sha1(plate.encode()).hexdigest()
    return "val" if int(digest[:8], 16) / 0xFFFFFFFF < holdout else "train"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--min-width", type=int, default=40,
                        help="skip boxes narrower than this; below it there is "
                             "nothing to read even in principle")
    parser.add_argument("--pad", type=float, default=0.06)
    args = parser.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    if not root.exists():
        print(f"dataset not found: {root}", file=sys.stderr)
        return 1
    for split in ("train", "val"):
        (out / split).mkdir(parents=True, exist_ok=True)

    handles = {s: (out / s / "labels.jsonl").open("w", encoding="utf-8")
               for s in ("train", "val")}
    counts = collections.Counter()
    widths: list[int] = []
    plates_in = collections.defaultdict(set)
    cache: dict[Path, object] = {}

    try:
        for i, (photo, bbox, plate) in enumerate(annotations(root)):
            image = cache.get(photo)
            if image is None:
                image = cv2.imread(str(photo))
                cache = {photo: image}          # one at a time; these are large
            if image is None:
                counts["unreadable photograph"] += 1
                continue

            x1, y1, x2, y2 = bbox
            if x2 - x1 < args.min_width:
                counts["too small"] += 1
                continue

            h, w = image.shape[:2]
            pad_x = int((x2 - x1) * args.pad)
            pad_y = int((y2 - y1) * args.pad)
            crop = image[max(0, y1 - pad_y):min(h, y2 + pad_y),
                         max(0, x1 - pad_x):min(w, x2 + pad_x)]
            if crop.size == 0:
                counts["empty crop"] += 1
                continue

            split = split_of(plate, args.holdout)
            name = f"{i:05d}_{plate}.png"
            if not cv2.imwrite(str(out / split / name), crop):
                counts["could not write"] += 1
                continue

            handles[split].write(json.dumps({
                "image": f"{split}/{name}",
                "plate": plate,
                "source": str(photo.relative_to(root)),
                "box": list(bbox),
                "width": x2 - x1,
            }) + "\n")
            counts[split] += 1
            widths.append(x2 - x1)
            plates_in[split].add(plate)
    finally:
        for handle in handles.values():
            handle.close()

    written = counts["train"] + counts["val"]
    print(f"source     : {root}")
    print(f"written    : {written} crops -> {out}")
    print(f"  train    : {counts['train']:>5} crops, {len(plates_in['train'])} distinct plates")
    print(f"  val      : {counts['val']:>5} crops, {len(plates_in['val'])} distinct plates")
    overlap = plates_in["train"] & plates_in["val"]
    print(f"  overlap  : {len(overlap)} plate(s) in both splits"
          + ("  <- would invalidate any result" if overlap else "  (split is clean)"))
    for reason in ("too small", "empty crop", "unreadable photograph", "could not write"):
        if counts[reason]:
            print(f"  skipped  : {counts[reason]} ({reason})")
    if widths:
        widths.sort()
        print()
        print(f"plate width: min {widths[0]} px, median {widths[len(widths)//2]} px, "
              f"max {widths[-1]} px")
        print("             Width is what decides whether a plate is readable at all,")
        print("             so a model trained here only speaks for this range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
