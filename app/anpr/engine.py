"""ANPR: reading the number plate off a detected vehicle.

Two things make this workable on a CPU that is already running detection:

**It only looks at vehicles, and only sometimes.** OCR on every vehicle on
every frame would halve the frame rate for no gain, because consecutive frames
show the same plate. Each track gets a bounded number of attempts, spaced out,
and stops entirely once it has a confident read.

**It votes.** A plate seen across several frames produces several reads, and
the one that appears most often is far more trustworthy than any single one.
A plate read once at 60% confidence is a guess; the same string read four
times is a number.

The OCR backend is optional and lazily loaded. Without it, IBVAP still detects
and localises plates and saves the crop as evidence - an operator can read it
even when the machine cannot.
"""

from __future__ import annotations

import queue
import threading
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .. import ui
from ..detection.appearance import body_box, describe
from ..detection.base import Detection
from ..detection.classes import is_vehicle
from .ledger import PlateLedger, Sighting
from . import plate as plate_finder
from . import readers
from .detector import PlateDetector
from . import text as plate_text


@dataclass(frozen=True)
class PlateReading:
    """The best plate number known for one tracked vehicle."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    # How many reads went into this number, including any that were outvoted
    # at individual positions. Not "how many said exactly this".
    reads: int = 1
    raw: str = ""

    @property
    def display(self) -> str:
        return plate_text.format_display(self.text)

    @property
    def state(self) -> str | None:
        return plate_text.state_of(self.text)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "display": self.display,
            "state": self.state,
            "confidence": round(self.confidence, 3),
            "reads": self.reads,
            "bbox": list(self.bbox),
            "raw": self.raw,
        }


@dataclass
class _TrackPlate:
    """Everything gathered about one vehicle's plate so far."""

    votes: Counter = field(default_factory=Counter)
    confidence: dict[str, float] = field(default_factory=dict)
    bbox: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    raw: dict[str, str] = field(default_factory=dict)
    attempts: int = 0
    last_attempt_frame: int = -999
    crop: np.ndarray | None = None
    best_area: int = 0
    best_sharpness: float = 0.0
    blurred_skips: int = 0
    # True once the plate model supplied the crop, which beats anything the
    # text search would pick.
    crop_from_model: bool = False

    def consensus(self, min_reads: int) -> tuple[str, int] | None:
        """Vote per character position, not per whole string.

        Real reads of one plate disagree in *different* places: the same plate
        came back as DL5CZ2581 and DL9CI2581, each with a single character
        wrong, and between them they carry the right answer. Whole-string
        voting throws that away and reports nothing; voting per position
        recovers it.

        Only strings of the same length are combined - a different length is a
        different reading, not a different opinion about the same one.
        """
        if not self.votes:
            return None

        by_length: dict[int, Counter] = {}
        for text, count in self.votes.items():
            by_length.setdefault(len(text), Counter())[text] += count

        length = max(by_length, key=lambda n: sum(by_length[n].values()))
        group = by_length[length]
        total = sum(group.values())
        if total < min_reads:
            return None

        merged = ""
        for position in range(length):
            column = Counter()
            for text, count in group.items():
                column[text[position]] += count

            ranked = column.most_common(2)
            # A tie is not a consensus. With two reads disagreeing at one
            # position, every character there has exactly one vote, and taking
            # the first is an arbitrary pick wearing a consensus costume -
            # which is the failure this whole stage exists to avoid.
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                return None
            merged += ranked[0][0]
        return merged, total

    def best(self, min_reads: int = 1) -> PlateReading | None:
        if not self.votes:
            return None

        agreed = self.consensus(min_reads)
        if agreed is None:
            return None
        text, count = agreed
        if count < min_reads:
            return None
        # The merged string may never have been read verbatim, so fall back to
        # the best-known values from whichever read contributed most.
        anchor = max(self.votes, key=lambda t: (len(t) == len(text), self.votes[t]))
        return PlateReading(
            text=text,
            confidence=self.confidence.get(text, self.confidence.get(anchor, 0.0)),
            bbox=self.bbox.get(text, self.bbox.get(anchor, (0, 0, 0, 0))),
            reads=count,
            raw=self.raw.get(text, self.raw.get(anchor, "")),
        )


class OCRUnavailable(Exception):
    """Raised once, then remembered, when no OCR backend can be loaded."""


@dataclass
class ANPREngine:
    """Locates and reads number plates on tracked vehicles."""

    enabled: bool = True
    # Plate OCR on real traffic returns low confidence even when correct -
    # 0.2-0.45 is normal. Raising this bar simply loses every real plate. What
    # protects the log instead is the format validator, which throws out
    # anything that cannot be a plate, and the requirement that several frames
    # agree before a number is trusted.
    min_confidence: float = 0.12
    max_attempts: int = 12          # per vehicle, then give up
    # Frames between attempts on one vehicle. A vehicle crossing a frame is
    # visible for tens of frames and only a handful of agreeing reads are
    # needed, so reading every third frame was buying nothing and costing the
    # whole frame budget.
    attempt_interval: int = 10
    # Below this the plate cannot carry enough pixels to be read at all, so
    # attempting it spends the frame budget on a guaranteed failure.
    min_vehicle_width: int = 180
    confident_reads: int = 3        # agreeing reads after which we stop trying
    # Nothing is reported until this many frames agree on the same number.
    # Real footage showed why: consecutive frames of one vehicle produced
    # DLSC22581 and 0L9C12581, and the repair step turned each into a
    # different, entirely plausible-looking plate. A single read is a guess
    # dressed up as a fact, and a wrong number in an evidence log is worse
    # than an admitted blank.
    min_agreeing_reads: int = 2
    languages: tuple[str, ...] = ("en",)
    # Which recogniser to hand the crop to. None picks the best one that will
    # actually load, which is measured, not assumed - see readers.py.
    ocr_backend: str | None = None
    reader_name: str = "not loaded"

    # A vehicle crossing the frame is only legible for part of that time. These
    # two gates spend the attempt budget on the frames worth reading.
    min_sharpness: float = 45.0     # below this the crop is motion-blurred
    growth_to_retry: float = 1.15   # re-read once the vehicle is 15% closer

    # Plate reading costs about 1.4 s per attempt on a CPU, which is far more
    # than a frame is worth. It is enrichment, not a gate - nothing downstream
    # waits on a plate - so a live pipeline runs it on its own thread and keeps
    # its frame rate.
    #
    # Measured, and it does NOT help on a CPU-only machine: moving the work to
    # a thread does not create cores, so detection and recognition end up
    # fighting over the same ones and both get slower. On this footage it cost
    # 5.4 fps -> 4.7 fps. It is kept because on a box with a GPU the two use
    # different silicon and the picture reverses - but it stays opt-in, and off
    # by default, because the common case is CPU.
    background: bool = False
    # Frames waiting to be read. Small on purpose - if reading falls behind,
    # the newest view of a vehicle is worth more than a backlog of old ones.
    queue_size: int = 2

    # Set by the fleet runner so cameras pool their reads. One camera's bad
    # angle is another camera's clear one.
    ledger: PlateLedger | None = None
    camera_id: str = "CAM-01"
    # A model trained on plates, when one is installed. Without it the engine
    # falls back to handing the whole vehicle to the text detector, which is
    # measurably worse at finding the plate but still works.
    plate_detector: PlateDetector | None = None

    _plates: dict[int, _TrackPlate] = field(default_factory=dict, init=False)
    _reader = None
    _reader_failed: bool = field(default=False, init=False)
    _frame: int = field(default=0, init=False)
    _clock: float = field(default=0.0, init=False)
    _ocr_failures: int = field(default=0, init=False)
    last_error: str | None = field(default=None, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _work: "queue.Queue | None" = field(default=None, init=False, repr=False)
    _worker: "threading.Thread | None" = field(default=None, init=False, repr=False)
    _stopping: bool = field(default=False, init=False)
    dropped: int = field(default=0, init=False)
    attempts_made: int = field(default=0, init=False)

    # ------------------------------------------------------------------ OCR

    def _ocr(self):
        """Load a recogniser on first use. Absence degrades, it does not crash.

        Which one is measured rather than assumed - see app/anpr/readers.py.
        """
        if self._reader is not None or self._reader_failed:
            return self._reader
        reader, problem = readers.build(self.ocr_backend)
        if reader is None:
            # Reported once: without OCR, IBVAP still localises plates and
            # saves the crop, so this degrades the feature rather than the run.
            self._reader_failed = True
            self._reader = None
            self.last_error = problem
            ui.warn(f"plate OCR unavailable, saving crops only: {problem}")
            return None
        self._reader = reader
        self.reader_name = reader.name
        return self._reader

    @property
    def ocr_failures(self) -> int:
        return self._ocr_failures

    @property
    def ocr_available(self) -> bool:
        return self._ocr() is not None

    # -------------------------------------------------------------- reading

    def tick(self, now: float | None = None) -> None:
        self._frame += 1
        if now is not None:
            self._clock = now

    def observe(self, frame: np.ndarray, detection: Detection) -> PlateReading | None:
        """Consider one detected vehicle. Returns the best reading so far.

        In background mode this never blocks: the work is queued and the
        caller gets whatever is already known about this vehicle.
        """
        if not self.enabled or frame is None or detection.track_id is None:
            return None
        if not is_vehicle(detection.label):
            return None

        with self._lock:
            state = self._plates.setdefault(detection.track_id, _TrackPlate())
            before = state.best(self.min_agreeing_reads)
            due = self._should_attempt(state, detection)
            if due and self.background:
                # Claim the slot now so the frame loop does not queue the same
                # vehicle again before the worker has started on it.
                state.last_attempt_frame = self._frame

        if due:
            if self.background:
                self._enqueue(frame, detection)
            else:
                with self._lock:
                    self._attempt(frame, detection, state)

        with self._lock:
            reading = state.best(self.min_agreeing_reads)
        if self.ledger is not None:
            self._share(frame, detection, reading, before)
        return reading

    # ----------------------------------------------------------- background

    def start(self) -> None:
        """Warm the OCR model and start the reader thread.

        Loading EasyOCR takes about five seconds. Doing it lazily meant the
        first vehicle to appear stalled the camera for that long, which is
        exactly the moment a border post can least afford it.
        """
        if not self.enabled or not self.background or self._worker is not None:
            return
        self._ocr()
        self._work = queue.Queue(maxsize=max(1, self.queue_size))
        self._stopping = False
        self._worker = threading.Thread(
            target=self._run, daemon=True, name=f"anpr-{self.camera_id}"
        )
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._worker is None:
            return
        self._stopping = True
        try:
            self._work.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=timeout)
        self._worker = None

    def _enqueue(self, frame: np.ndarray, detection: Detection) -> None:
        if self._worker is None:
            self.start()
        if self._work.full():
            # Reading is already behind; copying a whole frame to throw it away
            # is the one cost the frame loop should not be paying.
            self.dropped += 1
            return
        # The frame belongs to the caller and will be reused, so the worker
        # gets its own copy of just the vehicle.
        try:
            self._work.put_nowait((frame.copy(), detection, self._frame))
        except queue.Full:
            # Reading is behind. Drop the oldest waiting frame rather than the
            # newest: a fresher view of the vehicle is the more useful one.
            try:
                self._work.get_nowait()
                self._work.put_nowait((frame.copy(), detection, self._frame))
                self.dropped += 1
            except (queue.Empty, queue.Full):
                self.dropped += 1

    def _run(self) -> None:
        while not self._stopping:
            item = self._work.get()
            if item is None:
                break
            frame, detection, frame_no = item
            try:
                with self._lock:
                    state = self._plates.setdefault(detection.track_id, _TrackPlate())
                self._attempt(frame, detection, state, frame_no=frame_no)
            except Exception as exc:  # pragma: no cover - defensive
                self.last_error = str(exc)

    @property
    def queued(self) -> int:
        return self._work.qsize() if self._work is not None else 0

    def _share(self, frame, detection: Detection, reading, before) -> None:
        """Keep the fleet ledger current for this vehicle."""
        look = describe(frame, detection.bbox, region=body_box)
        if reading is None:
            # Nothing read here; remember how it looked so another camera can
            # hand the number back later.
            self.ledger.note_sighting(
                Sighting(
                    camera_id=self.camera_id,
                    track_id=detection.track_id,
                    at=self._clock,
                    appearance=look,
                    label=detection.label,
                )
            )
        elif before is None or before.text != reading.text or before.reads != reading.reads:
            self.ledger.record_read(
                camera_id=self.camera_id,
                track_id=detection.track_id,
                plate=reading.text,
                confidence=reading.confidence,
                at=self._clock,
                appearance=look,
            )

    def fleet_plate(self, track_id: int | None):
        """This vehicle's plate, including one another camera read for it."""
        if self.ledger is None:
            return None
        return self.ledger.resolve(self.camera_id, track_id)

    def _should_attempt(self, state: _TrackPlate, detection: Detection) -> bool:
        if state.attempts >= self.max_attempts:
            return False
        if (detection.bbox[2] - detection.bbox[0]) < self.min_vehicle_width:
            # Too far away for the plate to hold readable characters.
            return False
        if self._frame - state.last_attempt_frame < self.attempt_interval:
            return False

        best = state.best(self.min_agreeing_reads)
        if best and best.reads >= self.confident_reads:
            # Enough agreeing reads: stop spending CPU on a plate we know -
            # unless the vehicle has since come a good deal closer, in which
            # case a better view is worth one more look. A vehicle stopped at
            # a checkpoint never grows, so this must not gate ordinary voting.
            return detection.area >= state.best_area * self.growth_to_retry
        return True

    def _attempt(
        self,
        frame: np.ndarray,
        detection: Detection,
        state: _TrackPlate,
        frame_no: int | None = None,
    ) -> None:
        """Find the plate, read it, and record what it said.

        Localisation prefers a purpose-trained plate detector. Where none is
        installed the whole vehicle goes to the text detector instead - worse,
        because a vehicle carries a lot of text-like texture, but workable.
        """
        state.attempts += 1
        state.last_attempt_frame = self._frame
        self.attempts_made += 1

        reader = self._ocr()
        if reader is None:
            self._keep_fallback_crop(frame, detection, state)
            return

        located = self._locate_plate(
            frame, detection, self._frame if frame_no is None else frame_no
        )
        if located is not None:
            crop = plate_finder.crop(frame, located, pad=4)
            if crop is not None:
                focus = plate_finder.sharpness(crop)
                if state.crop is None or focus > state.best_sharpness:
                    state.crop, state.best_sharpness = crop, focus
                    state.crop_from_model = True

        region = self._region_for(frame, detection)
        if region is None:
            return
        x1, y1, x2, y2 = region

        scaled, scale = plate_finder.upscale_for_text(frame[y1:y2, x1:x2])
        results = self._detect_text(scaled)
        if not results:
            return

        def to_frame(box):
            xs = [p[0] / scale for p in box]
            ys = [p[1] / scale for p in box]
            return (
                x1 + int(min(xs)), y1 + int(min(ys)),
                x1 + int(max(xs)), y1 + int(max(ys)),
            )

        # Keep the sharpest text region seen, whatever it turned out to say. An
        # operator can read a plate the machine refused to commit to, and that
        # only works if the crop survives even when nothing validates.
        #
        # But only when there was no better way to look. Measured on two real
        # street clips: with the plate model installed and finding nothing on a
        # vehicle, this fallback saved the video's own watermark five times out
        # of five - "wildfilmsindia.com", "msindia.com". An overlay is rendered
        # digitally, so it is sharper than any plate seen through a lens at
        # distance and wins a sharpness contest every time. A crop filed as
        # evidence of a plate, showing a watermark, is a false record.
        #
        # A trained model finding no plate is evidence there is no visible
        # plate. Searching the vehicle for text after that does not recover
        # one; it only invents one. Filtering on what the text *said* was tried
        # and rejected - EasyOCR returns fragments like "E76" on genuine Indian
        # plates, so that discards the real evidence along with the false.
        searched_properly = (
            self.plate_detector is not None and self.plate_detector.available
        )
        if not state.crop_from_model and not searched_properly:
            for box, _, _ in results:
                crop = plate_finder.crop(frame, to_frame(box))
                if crop is None:
                    continue
                focus = plate_finder.sharpness(crop)
                if state.crop is None or focus > state.best_sharpness:
                    state.crop, state.best_sharpness = crop, focus

        for box, raw, confidence in self._plate_like(results):
            bbox = to_frame(box)
            crop = plate_finder.crop(frame, bbox)
            if crop is not None and plate_finder.sharpness(crop) < self.min_sharpness:
                state.blurred_skips += 1
                continue

            normalised = plate_text.normalise(raw)
            if not normalised or confidence < self.min_confidence:
                continue

            with self._lock:
                state.votes[normalised] += 1
                state.best_area = max(state.best_area, detection.area, 1)
            if confidence >= state.confidence.get(normalised, 0.0):
                state.confidence[normalised] = confidence
                state.bbox[normalised] = bbox
                state.raw[normalised] = plate_text.clean(raw)
            return

    def _locate_plate(self, frame: np.ndarray, detection: Detection, frame_no: int):
        """The plate's own box, when a plate model is installed.

        Used for the *evidence crop*, which is the part this genuinely fixes:
        without it the saved image was as likely to be a grille as a plate.
        """
        if self.plate_detector is None or not self.plate_detector.available:
            return None
        # Every vehicle in this frame shares one detection pass.
        plates = self.plate_detector.detect_on(
            frame, detection.bbox, cache_key=(self.camera_id, frame_no)
        )
        return plates[0].bbox if plates else None

    def _region_for(self, frame: np.ndarray, detection: Detection):
        """Where to run text recognition.

        Deliberately *not* the plate box. Tight crops were measured three
        separate ways - plate-model box, padded plate box, lower-vehicle crop -
        and every one of them read worse than handing over the whole vehicle.
        The recogniser was trained on scenes and uses the surroundings to
        decide what is text at all; cropping to the plate takes that away.
        """
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = plate_finder.read_region(detection.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 40 or y2 - y1 < 20:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _rows(results, tolerance: float = 0.6):
        """Group detected text into rows, top to bottom, left to right.

        Commercial plates carry their number on two lines. Sorting fragments by
        x alone - which is what reading a single-line plate needs - interleaves
        the two rows and produces a string that can never validate.
        """
        boxed = []
        for box, raw, confidence in results:
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            boxed.append(
                {
                    "top": min(ys),
                    "mid": (min(ys) + max(ys)) / 2,
                    "height": max(1.0, max(ys) - min(ys)),
                    "left": min(xs),
                    "box": box,
                    "text": plate_text.clean(raw),
                    "confidence": float(confidence),
                }
            )
        boxed.sort(key=lambda b: b["mid"])

        rows: list[list[dict]] = []
        for item in boxed:
            for row in rows:
                # Same row if the vertical centres are within a character height.
                if abs(row[0]["mid"] - item["mid"]) <= tolerance * item["height"]:
                    row.append(item)
                    break
            else:
                rows.append([item])
        for row in rows:
            row.sort(key=lambda b: b["left"])
        return rows

    @classmethod
    def _plate_like(cls, results):
        """Detected text that could be a plate, most promising first.

        Overlay watermarks and shop signs read as long, wide strings; a plate
        is short, mixes letters with digits, and may be spread over two rows.
        """
        rows = cls._rows(results)

        def plausible(text: str) -> bool:
            return (
                4 <= len(text) <= 13
                and any(c.isdigit() for c in text)
                and any(c.isalpha() for c in text)
            )

        candidates = []

        def add(box, text, confidence):
            if plausible(text):
                candidates.append((box, text, confidence))

        for row in rows:
            # Each fragment on its own, and the whole row joined.
            for item in row:
                add(item["box"], item["text"], item["confidence"])
            if len(row) > 1:
                joined = "".join(item["text"] for item in row)
                box = [p for item in row for p in item["box"]]
                add(box, joined, min(item["confidence"] for item in row))

        # A two-line plate reads as the upper row followed by the lower one.
        for i in range(len(rows) - 1):
            upper = "".join(item["text"] for item in rows[i])
            lower = "".join(item["text"] for item in rows[i + 1])
            box = [p for item in rows[i] + rows[i + 1] for p in item["box"]]
            confidence = min(
                item["confidence"] for item in rows[i] + rows[i + 1]
            )
            add(box, upper + lower, confidence)

        # Prefer strings the right length for a plate, then the most confident.
        candidates.sort(
            key=lambda c: (0 if 8 <= len(c[1]) <= 11 else 1, -c[2])
        )
        return candidates

    def _keep_fallback_crop(self, frame, detection: Detection, state: _TrackPlate) -> None:
        """With no OCR available, still save the best plate-shaped region."""
        for candidate in plate_finder.candidates(frame, detection.bbox):
            crop = plate_finder.crop(frame, candidate.bbox)
            if crop is None:
                continue
            focus = plate_finder.sharpness(crop)
            if state.crop is None or focus > state.best_sharpness:
                state.crop, state.best_sharpness = crop, focus
            return

    @property
    def unconfirmed(self) -> dict[int, list[tuple[str, int]]]:
        """Numbers seen but not yet agreed on, for diagnostics."""
        return {
            tid: state.votes.most_common(3)
            for tid, state in self._plates.items()
            if state.votes and not state.best(self.min_agreeing_reads)
        }

    def _detect_text(self, image: np.ndarray):
        """The single seam through which all OCR happens.

        Returns the detector's own (box, text, confidence) triples, so plate
        selection can use *where* each string was found and not only what it
        said.
        """
        reader = self._ocr()
        if reader is None:
            return []
        try:
            return reader.read(image)
        except Exception as exc:
            # A swallowed failure here disables plate reading for the rest of
            # the session in complete silence, which is how a broken OCR call
            # can look exactly like "no vehicle ever had a readable plate".
            self._ocr_failures += 1
            self.last_error = str(exc)
            if self._ocr_failures == 1:
                ui.warn(f"[{self.camera_id}] plate OCR failed: {exc}")
            return []

    # ------------------------------------------------------------ retrieval

    def best_for(self, track_id: int | None) -> PlateReading | None:
        if track_id is None:
            return None
        with self._lock:
            state = self._plates.get(track_id)
            return state.best(self.min_agreeing_reads) if state else None

    def crop_for(self, track_id: int | None) -> np.ndarray | None:
        if track_id is None:
            return None
        state = self._plates.get(track_id)
        return state.crop if state else None

    def forget(self, track_id: int) -> None:
        self._plates.pop(track_id, None)

    def reset(self) -> None:
        self._plates.clear()
        self._frame = 0
        self.attempts_made = 0
