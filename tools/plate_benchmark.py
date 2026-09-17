"""Measure plate reading against ground truth, before trying to improve it.

The README has said for a long time that plate *reading* does not work, on the
evidence of a few dozen frames watched by eye. That was enough to stop claiming
the feature, but it is not enough to fix it: "does not work" gives nothing to
improve against, and no way to tell whether a change helped or merely felt
better.

This turns that judgement into a number. Every annotated plate is cropped from
its photograph at the annotated box and handed to the reader, so localisation
is taken out of the picture entirely - what is measured is recognition alone,
under conditions kinder than any camera will ever provide.

Three numbers come out, and the gap between them is the useful part:

* **exact**      - the whole plate, right. The only one that can be acted on.
* **character**  - how much of each plate is right, position by position. A
                   recogniser at 85% characters and 5% exact is close; one at
                   30% characters is not reading at all.
* **validated**  - how often the pipeline's own format check accepted the
                   answer, right or wrong. A validated *wrong* plate is the
                   dangerous case: it reaches the registry and mismatches
                   against an innocent vehicle.

    python tools/plate_benchmark.py
    python tools/plate_benchmark.py --limit 200 --report data/plate-benchmark.json

Run it before and after any change to reading. A change that does not move
these numbers did not work, whatever it looked like on one video.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.anpr import normalise  # noqa: E402
from app.anpr.engine import ANPREngine  # noqa: E402

DEFAULT_ROOT = Path("data/datasets/indian-plates")


@dataclass
class Sample:
    image: Path
    text: str
    box: tuple[int, int, int, int]


@dataclass
class Score:
    total: int = 0
    exact: int = 0
    validated: int = 0
    validated_wrong: int = 0
    empty: int = 0
    char_hit: int = 0
    char_total: int = 0
    wrong: list[tuple[str, str]] = field(default_factory=list)

    @property
    def exact_pct(self) -> float:
        return 100.0 * self.exact / self.total if self.total else 0.0

    @property
    def char_pct(self) -> float:
        return 100.0 * self.char_hit / self.char_total if self.char_total else 0.0


def character_overlap(truth: str, guess: str) -> tuple[int, int]:
    """How many characters are right, aligned from the left, then from the right.

    Plates lose characters at either end - a bumper crops the left, a tow bar
    the right - so scoring one alignment only would call a correct read with a
    missing first character almost entirely wrong.
    """
    if not truth:
        return 0, 0
    best = 0
    for a, b in ((truth, guess), (truth[::-1], guess[::-1])):
        hit = sum(1 for x, y in zip(a, b) if x == y)
        best = max(best, hit)
    return best, len(truth)


def load(root: Path) -> list[Sample]:
    """Every annotation whose photograph is actually present."""
    samples: list[Sample] = []
    for xml in root.rglob("*.xml"):
        try:
            tree = ET.parse(xml)
        except ET.ParseError:
            continue
        name = tree.findtext("filename") or ""
        image = xml.with_suffix(".jpg")
        for candidate in (xml.parent / name, image, xml.with_suffix(".png"),
                          xml.with_suffix(".jpeg")):
            if candidate.exists():
                image = candidate
                break
        else:
            continue

        for obj in tree.findall("object"):
            text = (obj.findtext("name") or "").strip().upper()
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
            samples.append(Sample(image, text, bbox))
    return samples


def read_plate(engine: ANPREngine, crop) -> tuple[str, bool]:
    """The pipeline's own reading path, on an already-located plate.

    Returns what it said and whether its format check accepted it, so a
    confidently wrong answer can be counted separately from an honest blank.
    """
    from app.anpr import plate as plate_finder

    scaled, _ = plate_finder.upscale_for_text(crop)
    results = engine._detect_text(scaled)
    if not results:
        return "", False

    for _, raw, _ in engine._plate_like(results):
        cleaned = normalise(raw)
        if cleaned:
            return cleaned, True

    # Nothing passed the plate-shape filter; report the raw text anyway so the
    # character score can show whether the recogniser was close or nowhere.
    joined = "".join(str(raw) for _, raw, _ in results)
    return "".join(ch for ch in joined.upper() if ch.isalnum()), False


def consensus_run(engine: ANPREngine, samples: list[Sample], args) -> int:
    """Score the way the pipeline actually decides, not one read at a time.

    The pipeline never acts on a single read. It gathers several across frames
    and votes per character position, and only reports a number once enough
    reads agree. Scoring single reads therefore measures something the system
    does not do - and understates it badly, because independent errors cancel.

    This dataset happens to contain several photographs of the same plate, so
    that can be measured rather than assumed: each group of photographs stands
    in for the frames of one pass.

    The number that matters here is precision. A plate reported wrong reaches
    the vehicle registry and flags an innocent driver, so a system that reports
    fewer numbers and is right about them beats one that reports more.
    """
    from collections import defaultdict

    from app.anpr.engine import _TrackPlate

    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups["".join(ch for ch in sample.text if ch.isalnum())].append(sample)
    usable = {k: v[:args.consensus] for k, v in groups.items() if len(v) >= 2}

    print(f"consensus: {len(usable)} plates with 2+ photographs, "
          f"up to {args.consensus} reads each")
    print(f"           reporting only when {args.min_agreeing} reads agree")
    print()

    reported = correct = silent = 0
    wrong: list[tuple[str, str]] = []
    started = time.time()

    for i, (truth, group) in enumerate(sorted(usable.items()), 1):
        state = _TrackPlate()
        for sample in group:
            image = cv2.imread(str(sample.image))
            if image is None:
                continue
            x1, y1, x2, y2 = sample.box
            h, w = image.shape[:2]
            pad = max(2, (x2 - x1) // 20)
            crop = image[max(0, y1 - pad):min(h, y2 + pad),
                         max(0, x1 - pad):min(w, x2 + pad)]
            if crop.size == 0:
                continue
            guess, validated = read_plate(engine, crop)
            if validated and guess:
                state.votes[guess] += 1

        settled = state.consensus(args.min_agreeing)
        if settled is None:
            silent += 1
        else:
            reported += 1
            if settled[0] == truth:
                correct += 1
            elif len(wrong) < 200:
                wrong.append((truth, settled[0]))
        if i % 50 == 0:
            print(f"  {i}/{len(usable)}  reported {reported}  correct {correct}")

    elapsed = time.time() - started
    precision = 100.0 * correct / reported if reported else 0.0
    recall = 100.0 * correct / len(usable) if usable else 0.0
    print()
    print("=" * 62)
    print(f"  plates              : {len(usable)}")
    print(f"  reported a number   : {reported:>5}  "
          f"({100.0 * reported / max(len(usable), 1):5.1f}%)")
    print(f"  ...and were RIGHT   : {correct:>5}")
    print(f"  ...and were WRONG   : {reported - correct:>5}   <- reaches the registry")
    print(f"  said nothing        : {silent:>5}")
    print()
    print(f"  PRECISION           : {precision:5.1f}%   of what it reports, this is right")
    print(f"  recall              : {recall:5.1f}%   of all plates, this many were read")
    print(f"  time                : {elapsed:.0f}s")
    print("=" * 62)

    if args.show_wrong and wrong:
        print()
        print(f"  {'truth':<14}{'reported':<20}")
        for truth, guess in wrong[:args.show_wrong]:
            print(f"  {truth:<14}{guess:<20}")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "mode": "consensus",
            "min_agreeing": args.min_agreeing,
            "max_reads": args.consensus,
            "plates": len(usable),
            "reported": reported,
            "correct": correct,
            "wrong": reported - correct,
            "precision_pct": round(precision, 2),
            "recall_pct": round(recall, 2),
            "wrong_examples": wrong,
        }, indent=2), encoding="utf-8")
        print()
        print(f"  written: {args.report}")
    return 0


def use_paddle(engine: ANPREngine) -> None:
    """Swap the recogniser and change nothing else.

    Only `_detect_text` is replaced - the single seam through which all OCR
    happens. The plate-shape filter, the format repair, the repair cap and the
    consensus vote all stay exactly as they are, so a difference in the result
    is a difference in the recogniser and not in anything around it.
    """
    from paddleocr import PaddleOCR

    # oneDNN's kernels are not implemented for this build on Windows and abort
    # at inference; without it the same model runs, a little slower.
    reader = PaddleOCR(use_textline_orientation=False, lang="en",
                       enable_mkldnn=False)

    def detect(self, image, localised=True):
        try:
            pages = reader.predict(image)
        except Exception:
            return []
        out = []
        for page in pages:
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            boxes = page.get("rec_polys") or page.get("dt_polys") or []
            for i, text in enumerate(texts):
                box = boxes[i] if i < len(boxes) else [[0, 0], [1, 0], [1, 1], [0, 1]]
                box = [[float(x), float(y)] for x, y in box]
                score = float(scores[i]) if i < len(scores) else 0.5
                out.append((box, str(text), score))
        return out

    ANPREngine._detect_text = detect


def use_paddle_rec(engine: ANPREngine) -> None:
    """PaddleOCR's recogniser alone, with its text-detection stage skipped.

    The plate is already located by the time this runs, so paying for text
    detection again is paying to find what we have already found. Skipping it
    measured 0.219s a plate against 5.59s for the full PaddleOCR pipeline and
    0.56s for EasyOCR - so this is both the most accurate recogniser tried and
    the fastest.

    The cost is that a two-line plate arrives as one image with no way to tell
    the rows apart, where the full pipeline would have returned each line
    separately. Whether that matters is what the benchmark is for.
    """
    from paddleocr import TextRecognition

    reader = TextRecognition()

    def detect(self, image, localised=True):
        try:
            results = reader.predict(image)
        except Exception:
            return []
        h, w = image.shape[:2]
        whole = [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]
        out = []
        for item in results or []:
            text = item.get("rec_text") if isinstance(item, dict) else None
            if not text:
                continue
            score = item.get("rec_score", 0.5) if isinstance(item, dict) else 0.5
            out.append((whole, str(text), float(score)))
        return out

    ANPREngine._detect_text = detect


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--limit", type=int, default=0, help="0 = every sample")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", default=None, help="write the full result as JSON")
    parser.add_argument("--show-wrong", type=int, default=12)
    parser.add_argument(
        "--consensus", type=int, default=0, metavar="N",
        help="read each plate from N different photographs and combine them the "
             "way the pipeline does, instead of scoring single reads",
    )
    parser.add_argument("--min-agreeing", type=int, default=2)
    parser.add_argument("--engine", choices=["easyocr", "paddle", "paddle-rec"], default="easyocr",
                        help="which recogniser to measure; everything downstream "
                             "of it - plate-shape filter, format repair, consensus "
                             "- stays identical, so the comparison is like for like")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"dataset not found: {root}", file=sys.stderr)
        return 1

    samples = load(root)
    if not samples:
        print(f"no annotated plates under {root}", file=sys.stderr)
        return 1

    if args.limit and args.limit < len(samples):
        random.Random(args.seed).shuffle(samples)
        samples = samples[:args.limit]

    engine = ANPREngine()
    if args.engine == "paddle":
        use_paddle(engine)
    elif args.engine == "paddle-rec":
        use_paddle_rec(engine)
    elif engine._ocr() is None:
        print("OCR unavailable - install easyocr first", file=sys.stderr)
        return 1

    print(f"dataset  : {root}")
    print(f"engine   : {args.engine}")
    print(f"samples  : {len(samples)} annotated plates")
    print("reading  : the pipeline's own path, on the annotated box")
    print("           (localisation removed - this measures recognition alone)")
    print()

    if args.consensus:
        return consensus_run(engine, samples, args)

    score = Score()
    started = time.time()
    for i, sample in enumerate(samples, 1):
        image = cv2.imread(str(sample.image))
        if image is None:
            continue
        x1, y1, x2, y2 = sample.box
        h, w = image.shape[:2]
        pad = max(2, (x2 - x1) // 20)
        crop = image[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size == 0:
            continue

        guess, validated = read_plate(engine, crop)
        truth = "".join(ch for ch in sample.text if ch.isalnum())

        score.total += 1
        hit, total = character_overlap(truth, guess)
        score.char_hit += hit
        score.char_total += total
        if not guess:
            score.empty += 1
        if guess == truth:
            score.exact += 1
        elif len(score.wrong) < 400:
            score.wrong.append((truth, guess))
        if validated:
            score.validated += 1
            if guess != truth:
                score.validated_wrong += 1

        if i % 100 == 0:
            print(f"  {i}/{len(samples)}  exact {score.exact_pct:5.1f}%  "
                  f"chars {score.char_pct:5.1f}%")

    elapsed = time.time() - started
    print()
    print("=" * 62)
    print(f"  samples read       : {score.total}")
    print(f"  EXACT match        : {score.exact:>5}  ({score.exact_pct:5.1f}%)")
    print(f"  character accuracy : {score.char_hit}/{score.char_total}  "
          f"({score.char_pct:5.1f}%)")
    print(f"  returned nothing   : {score.empty:>5}  "
          f"({100.0 * score.empty / max(score.total, 1):5.1f}%)")
    print(f"  passed format check: {score.validated:>5}  "
          f"({100.0 * score.validated / max(score.total, 1):5.1f}%)")
    print(f"  ...and were WRONG  : {score.validated_wrong:>5}   <- reaches the registry")
    print(f"  time               : {elapsed:.0f}s "
          f"({elapsed / max(score.total, 1):.2f}s a plate)")
    print("=" * 62)

    if args.show_wrong and score.wrong:
        print()
        print(f"  {'truth':<14}{'read as':<20}")
        for truth, guess in score.wrong[:args.show_wrong]:
            # A recogniser trained on more than Latin script can return a
            # character this console cannot encode. Crashing at the point of
            # *printing the results* would throw away the whole run, so the
            # unprintable characters are shown as escapes instead.
            line = f"  {truth:<14}{guess or '(nothing)':<20}"
            print(line.encode(sys.stdout.encoding or "utf-8",
                              errors="backslashreplace").decode(
                                  sys.stdout.encoding or "utf-8"))

    if args.report:
        Path(args.report).write_text(json.dumps({
            "dataset": str(root),
            "samples": score.total,
            "exact": score.exact,
            "exact_pct": round(score.exact_pct, 2),
            "character_pct": round(score.char_pct, 2),
            "empty": score.empty,
            "validated": score.validated,
            "validated_wrong": score.validated_wrong,
            "seconds": round(elapsed, 1),
            "wrong_examples": score.wrong[:200],
        }, indent=2), encoding="utf-8")
        print(f"\n  written: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
