"""Deciding where on a vehicle to look for a number plate.

The obvious approach is here but demoted, because it was measured and it lost.

Searching for a bright, high-contrast, plate-shaped rectangle with contours is
cheap and needs no second model, and on a clean frontal photograph it works
perfectly. On real traffic it does not: a vehicle carries many plate-shaped
rectangles - grille slats, bumper shadows, badge recesses - and shape alone
cannot tell them from the plate. Across 186 vehicle detections it produced 43
"plates", every one of them wrong. Adding `character_count`, which asks whether
a region contains glyphs at all, cut that to 7 - still all wrong.

So `candidates` survives only as the fallback for when no OCR backend is
available, where a plausible crop is better than nothing. The main path uses
`read_region` to hand a whole vehicle to a trained text detector, which can
tell text from texture in a way these heuristics cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Indian plates are roughly 2:1 (single row) or 4.5:1 for the long format.
MIN_ASPECT = 1.8
MAX_ASPECT = 6.0

# A plate is a small part of the vehicle, but not a speck of noise.
MIN_AREA_FRACTION = 0.004
MAX_AREA_FRACTION = 0.25

MIN_PLATE_WIDTH = 55
MIN_PLATE_HEIGHT = 16

# A plate carries 6-10 characters in a row. Counting them is what separates a
# plate from the many other plate-shaped rectangles on a vehicle - a grille
# slat, a bumper shadow, a badge recess - which was the failure mode on real
# traffic footage: every one of those scored well on shape alone.
MIN_CHARACTERS = 4
# A found plate box usually includes some border and margin, so the glyphs
# occupy less of it than the plate's own proportions would suggest.
CHAR_MIN_HEIGHT_FRACTION = 0.22     # of the candidate's height
CHAR_MAX_HEIGHT_FRACTION = 0.95
CHAR_MAX_WIDTH_FRACTION = 0.28      # of the candidate's width


@dataclass(frozen=True)
class PlateCandidate:
    """A region that looks like a plate, in full-frame coordinates."""

    bbox: tuple[int, int, int, int]
    score: float
    characters: int = 0

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


def character_count(region: np.ndarray) -> int:
    """How many character-shaped blobs sit in a row inside this region.

    Plate characters are tall, narrow, well separated and all about the same
    height. Almost nothing else on a vehicle produces six of those in a line.
    """
    if region is None or region.size == 0:
        return 0
    grey = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    height, width = grey.shape[:2]
    if height < 10 or width < 30:
        return 0

    # Plates are dark-on-light or light-on-dark; try both polarities.
    best = 0
    for invert in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, binary = cv2.threshold(grey, 0, 255, invert | cv2.THRESH_OTSU)
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        found = 0
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if not (CHAR_MIN_HEIGHT_FRACTION * height <= h <= CHAR_MAX_HEIGHT_FRACTION * height):
                continue
            if w > CHAR_MAX_WIDTH_FRACTION * width or w < 2:
                continue
            if area < 0.12 * w * h:      # too sparse to be a glyph
                continue
            if area > 0.92 * w * h:      # a solid block, not a character
                continue
            found += 1
        best = max(best, found)
    return best


def _search_region(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The lower half of the vehicle, where a plate actually is."""
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    return x1, y1 + int(height * 0.40), x2, y2


def candidates(
    frame: np.ndarray, vehicle_bbox: tuple[int, int, int, int], limit: int = 3
) -> list[PlateCandidate]:
    """Plate-shaped regions inside a vehicle box, best first."""
    if frame is None or frame.size == 0:
        return []

    height, width = frame.shape[:2]
    rx1, ry1, rx2, ry2 = _search_region(vehicle_bbox)
    rx1, ry1 = max(0, rx1), max(0, ry1)
    rx2, ry2 = min(width, rx2), min(height, ry2)
    if rx2 - rx1 < MIN_PLATE_WIDTH or ry2 - ry1 < MIN_PLATE_HEIGHT:
        return []

    region = frame[ry1:ry2, rx1:rx2]
    grey = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    grey = cv2.bilateralFilter(grey, 7, 60, 60)

    # A plate is a dense band of vertical strokes, so emphasise vertical edges
    # and then close them into one blob.
    edges = cv2.Sobel(grey, cv2.CV_8U, 1, 0, ksize=3)
    _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region_area = float((rx2 - rx1) * (ry2 - ry1))

    found: list[PlateCandidate] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < MIN_PLATE_WIDTH or h < MIN_PLATE_HEIGHT:
            continue
        aspect = w / float(h)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        fraction = (w * h) / region_area
        if not (MIN_AREA_FRACTION <= fraction <= MAX_AREA_FRACTION):
            continue

        patch = region[y:y + h, x:x + w]
        glyphs = character_count(patch)
        if glyphs < MIN_CHARACTERS:
            # Plate-shaped but carrying no text: a grille, a shadow, a badge.
            continue

        # A plate is also brighter than the bodywork around it.
        brightness = float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean()) / 255.0
        fill = cv2.contourArea(contour) / float(w * h)
        depth = (y + h / 2) / float(ry2 - ry1)
        score = (
            min(1.0, glyphs / 8.0) * 0.45
            + brightness * 0.20
            + fill * 0.15
            + depth * 0.10
            + min(1.0, fraction * 8) * 0.10
        )
        found.append(
            PlateCandidate(
                bbox=(rx1 + x, ry1 + y, rx1 + x + w, ry1 + y + h),
                score=score,
                characters=glyphs,
            )
        )

    found.sort(key=lambda c: -c.score)
    return found[:limit]


def read_region(bbox: tuple[int, int, int, int], lower: float = 0.0) -> tuple[int, int, int, int]:
    """The part of a vehicle worth handing to a text detector.

    Contour heuristics were tried first and failed on real traffic: a vehicle
    carries many plate-shaped rectangles - grille slats, bumper shadows, badge
    recesses - and shape alone cannot tell them from the plate. A trained text
    detector can, so the job here is only to narrow it to where a plate is.
    """
    x1, y1, x2, y2 = bbox
    return x1, y1 + int((y2 - y1) * lower), x2, y2


def upscale_for_text(region: np.ndarray, min_width: int = 320) -> tuple[np.ndarray, float]:
    """Enlarge a small region so the detector has pixels to work with."""
    height, width = region.shape[:2]
    if width >= min_width or width == 0:
        return region, 1.0
    scale = min_width / width
    return (
        cv2.resize(region, (min_width, int(height * scale)), interpolation=cv2.INTER_CUBIC),
        scale,
    )


def crop(frame: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 3) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1 = max(0, bbox[0] - pad)
    y1 = max(0, bbox[1] - pad)
    x2 = min(width, bbox[2] + pad)
    y2 = min(height, bbox[3] + pad)
    if x2 - x1 < 8 or y2 - y1 < 6:
        return None
    return frame[y1:y2, x1:x2].copy()


def sharpness(crop: np.ndarray) -> float:
    """How much detail a crop actually carries.

    A vehicle moving past a camera is blurred in most frames, and OCR on a
    blurred plate does not fail quietly - it returns confident nonsense. The
    variance of the Laplacian is the cheapest usable measure of focus.
    """
    if crop is None or crop.size == 0:
        return 0.0
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


# Enhancement before OCR was tried and measurably HURT, so it is deliberately
# absent. Cropping tight to the plate and then upscaling, denoising, CLAHE-ing
# and unsharp-masking it turned a read of "DLSC22581" at 0.46 confidence into
# "UOUOTUU4IEDLIC72581" at 0.01 on the same frame: the upscale amplifies JPEG
# artefacts, and the padding pulls in bumper texture that then reads as
# characters. The recogniser does better on the region at its natural
# resolution, with the surrounding context it was trained to use.
