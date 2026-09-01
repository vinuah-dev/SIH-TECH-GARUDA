"""The recognisers a plate crop can be handed to.

Three were measured against 400 annotated Indian plates, each cropped to its
annotated box so that localisation played no part and only recognition was
being judged:

| recogniser | whole plate right | characters right | seconds a plate |
| --- | --- | --- | --- |
| EasyOCR                  | 34.5% | 71.2% | 0.56 |
| PaddleOCR, full pipeline | 60.2% | 80.3% | 5.59 |
| **PaddleOCR, recogniser only** | **73.2%** | **90.7%** | **0.22** |

The last row is both the most accurate and the fastest, which looks too good
until the reason is clear: **the plate is already located by the time any of
this runs.** PaddleOCR's full pipeline pays to find text all over again, and
worse, its text detector often splits one plate into several fragments that then
have to be stitched back together. Handing the recogniser the whole crop skips
the cost and the fragmentation together.

What that trades away is row separation. A two-line plate arrives as one image
with nothing to say where the first line ends - the full pipeline would have
returned each line separately. The measurement above includes two-line plates
and still comes out ahead, so the trade is worth taking; it is not free.

A missing recogniser degrades the feature rather than the run: plates are still
located and their crops still saved, and an operator can read a crop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


class Reader(Protocol):
    """Turns an image into (box, text, confidence) triples.

    The box matters: plate selection uses *where* each string was found, not
    only what it said, so that a two-line plate can be reassembled in the right
    order and a watermark across the bonnet can be told from a plate.
    """

    name: str

    def read(self, image: np.ndarray) -> Sequence[tuple]:
        ...


@dataclass
class EasyOCRReader:
    """General scene-text recognition. Slower and much less accurate here."""

    name: str = "EasyOCR"
    languages: tuple[str, ...] = ("en",)
    _reader: object | None = field(default=None, init=False, repr=False)

    def load(self) -> None:
        import easyocr

        self._reader = easyocr.Reader(list(self.languages), gpu=False, verbose=False)

    def read(self, image: np.ndarray) -> Sequence[tuple]:
        return self._reader.readtext(image, detail=1, paragraph=False)


@dataclass
class PaddleRecogniser:
    """PaddleOCR's recogniser with its text-detection stage skipped.

    The measured default. Because detection is skipped, every string comes back
    against the whole crop rather than its own box - so a two-line plate is one
    result, not two.
    """

    name: str = "PaddleOCR (recogniser only)"
    _reader: object | None = field(default=None, init=False, repr=False)

    def load(self) -> None:
        from paddleocr import TextRecognition

        self._reader = TextRecognition()

    def read(self, image: np.ndarray) -> Sequence[tuple]:
        results = self._reader.predict(image)
        height, width = image.shape[:2]
        whole = [[0.0, 0.0], [float(width), 0.0],
                 [float(width), float(height)], [0.0, float(height)]]

        out = []
        for item in results or []:
            if not isinstance(item, dict):
                continue
            text = item.get("rec_text")
            if not text:
                continue
            out.append((whole, str(text), float(item.get("rec_score", 0.5))))
        return out


@dataclass
class PaddleOCRReader:
    """PaddleOCR's full pipeline: its own text detection, then recognition.

    Kept because it is the only backend that separates the rows of a two-line
    plate on its own. It is 25x the cost of the recogniser alone and measured
    *less* accurate on this data, so it is not the default.
    """

    name: str = "PaddleOCR (full)"
    _reader: object | None = field(default=None, init=False, repr=False)

    def load(self) -> None:
        from paddleocr import PaddleOCR

        # oneDNN's kernels are unimplemented for this build on Windows and
        # abort at inference. Without it the same model runs, a little slower.
        self._reader = PaddleOCR(use_textline_orientation=False, lang="en",
                                 enable_mkldnn=False)

    def read(self, image: np.ndarray) -> Sequence[tuple]:
        out = []
        for page in self._reader.predict(image) or []:
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            boxes = page.get("rec_polys") or page.get("dt_polys") or []
            for i, text in enumerate(texts):
                if not text:
                    continue
                box = boxes[i] if i < len(boxes) else [[0, 0], [1, 0], [1, 1], [0, 1]]
                out.append((
                    [[float(x), float(y)] for x, y in box],
                    str(text),
                    float(scores[i]) if i < len(scores) else 0.5,
                ))
        return out


BACKENDS = {
    "paddle-rec": PaddleRecogniser,
    "paddle": PaddleOCRReader,
    "easyocr": EasyOCRReader,
}

# Tried in order when nothing is asked for by name. The best measured backend
# first, and a general recogniser behind it so a machine without PaddleOCR
# still reads plates rather than silently reading none.
PREFERENCE = ("paddle-rec", "easyocr")


def build(name: str | None = None) -> tuple[object | None, str | None]:
    """The requested recogniser, or the best one that will actually load.

    Returns the reader and, when nothing loaded, why - so the caller can say
    so once instead of failing plate by plate in silence.
    """
    if name:
        backend = BACKENDS.get(name)
        if backend is None:
            return None, f"unknown OCR backend {name!r}; try {', '.join(BACKENDS)}"
        reader = backend()
        try:
            reader.load()
        except Exception as exc:
            return None, f"{reader.name} unavailable: {exc}"
        return reader, None

    problems = []
    for candidate in PREFERENCE:
        reader = BACKENDS[candidate]()
        try:
            reader.load()
            return reader, None
        except Exception as exc:
            problems.append(f"{candidate}: {exc}")
    return None, "; ".join(problems)
