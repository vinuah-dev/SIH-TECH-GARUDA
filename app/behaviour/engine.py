"""Behaviour Engine.

Turns the Context Engine's raw measurements into **named behaviours** an
operator would recognise:

    dwell 18s inside RESTRICTED   ->  LOITERING
    approaching for 12 frames     ->  BORDER-FACING MOVEMENT
    heading variance 0.63         ->  ERRATIC MOVEMENT
    moving during the night window ->  NIGHT MOVEMENT
    2.4 m/s on a calibrated camera ->  RAPID MOVEMENT
    person filling 70% of the frame ->  CAMERA TAMPERING

The Context Engine measures, this stage classifies, and the Risk Engine scores
whatever this stage names. Keeping the three apart is what makes an alert
explainable: the operator sees the measurement, the behaviour it triggered and
the points it contributed.

Every threshold here is a deliberate policy choice, not a learned one. Anything
requiring a learned baseline of "normal" for a given camera is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..context.engine import TrackContext
from ..detection.classes import is_person

LOITERING = "LOITERING"
BORDER_FACING = "BORDER-FACING MOVEMENT"
ERRATIC = "ERRATIC MOVEMENT"
NIGHT_MOVEMENT = "NIGHT MOVEMENT"
RUNNING = "RAPID MOVEMENT"
APPROACHING_CAMERA = "APPROACHING CAMERA"
CAMERA_TAMPERING = "CAMERA TAMPERING"
PLATE_MISMATCH = "PLATE MISMATCH"
WATCHLISTED = "WATCHLISTED VEHICLE"
UNREGISTERED = "UNREGISTERED PLATE"
AUTHORISED = "AUTHORISED PERSONNEL"
UNRECOGNISED = "UNRECOGNISED FACE"


@dataclass(frozen=True)
class Behaviour:
    """One named behaviour detected for one track at one instant."""

    name: str
    points: int
    detail: str

    def __str__(self) -> str:
        return f"{self.name} ({self.detail})"


@dataclass
class BehaviourEngine:
    """Classifies per-track movement context into named behaviours."""

    # LOITERING: standing inside a zone. Graded, because 60 seconds is not the
    # same as 6, and it must be able to outweigh a single-shot signal.
    loiter_after: float = 5.0
    loiter_per_second: float = 1.5
    loiter_cap: int = 20

    # BORDER-FACING: a single frame of "approaching" is noise. Requiring it to
    # persist is the whole point of putting this behind the Context Engine.
    approach_frames: int = 5
    border_facing_points: int = 8

    # ERRATIC: pacing, doubling back, circling.
    erratic_threshold: float = 0.45
    erratic_min_samples: int = 8
    erratic_points: int = 6

    # NIGHT MOVEMENT: active movement in the dark, on top of the flat
    # night-time context factor for merely being present.
    night_movement_points: int = 4

    # RAPID MOVEMENT: only meaningful on a calibrated camera, because "fast"
    # has no meaning in frame widths per second - the same figure means
    # different things at different camera angles.
    # A plate that does not match the vehicle carrying it is the signature of
    # a cloned or transplanted plate - a standard way to move a vehicle that
    # would otherwise be stopped. It outranks anything about how it is driven.
    plate_mismatch_points: int = 35
    watchlisted_points: int = 45
    # Absence from an extract is weak evidence: the extract may simply be
    # incomplete or out of date, so this only nudges the score.
    unregistered_points: int = 5

    # Recognising the guard is what lets the other alert mean something, so
    # this subtracts enough to take a routine patrol out of alerting range.
    authorised_points: int = -45
    # Not being on the roster is the normal case for almost everybody at a
    # border, so by default it counts for nothing. A checkpoint where everyone
    # genuinely should be enrolled can raise it.
    unrecognised_points: int = 0

    running_speed_mps: float = 2.0
    # A vehicle at 2 m/s is crawling; the threshold has to be its own.
    vehicle_speed_mps: float = 11.0
    running_points: int = 6

    # CAMERA TAMPERING: surveillance cameras are mounted high and look outward,
    # so nobody has a legitimate reason to be close enough to fill the frame.
    # Someone at arm's length from the lens is reaching for the camera, and
    # blinding the sensor defeats every other behaviour in this file - which is
    # why this outweighs all of them.
    tamper_height_ratio: float = 0.65
    tamper_points: int = 30

    # Getting rapidly larger means closing on the camera along its own axis,
    # which a ground-plane position cannot see at all.
    camera_approach_growth: float = 0.06   # height fraction gained per second
    camera_approach_min_ratio: float = 0.35
    camera_approach_points: int = 10

    _approach_streak: dict[int, int] = field(default_factory=dict, init=False)

    def classify(self, track: TrackContext, label: str = "PERSON") -> list[Behaviour]:
        """Name every behaviour currently exhibited by one track.

        `label` decides which behaviours even apply: reaching for the camera is
        something a person does, and "fast" means something different to a car.
        """
        behaviours: list[Behaviour] = []

        loitering = self._loitering(track)
        if loitering:
            behaviours.append(loitering)

        border_facing = self._border_facing(track)
        if border_facing:
            behaviours.append(border_facing)

        erratic = self._erratic(track)
        if erratic:
            behaviours.append(erratic)

        night = self._night_movement(track)
        if night:
            behaviours.append(night)

        running = self._running(track, label)
        if running:
            behaviours.append(running)

        # A truck filling the frame is a truck driving past, not tampering.
        tampering = self._camera_tampering(track) if is_person(label) else None
        if tampering:
            behaviours.append(tampering)
        elif is_person(label) and (approaching_camera := self._approaching_camera(track)):
            # Once tampering is called, "approaching" is redundant noise.
            behaviours.append(approaching_camera)

        return behaviours

    # ------------------------------------------------------------ behaviours

    def _loitering(self, track: TrackContext) -> Behaviour | None:
        if not track.inside_zone:
            return None
        excess = track.dwell_seconds - self.loiter_after
        if excess <= 0:
            return None
        points = min(self.loiter_cap, int(round(excess * self.loiter_per_second)))
        if points <= 0:
            return None
        return Behaviour(
            name=LOITERING,
            points=points,
            detail=f"{track.dwell_seconds:.0f}s continuously inside {track.zone.name}",
        )

    def _border_facing(self, track: TrackContext) -> Behaviour | None:
        """Sustained approach toward an armed zone."""
        streak = self._approach_streak.get(track.track_id, 0)
        if track.approaching and track.is_moving:
            streak += 1
        else:
            streak = 0
        self._approach_streak[track.track_id] = streak

        if streak < self.approach_frames:
            return None
        return Behaviour(
            name=BORDER_FACING,
            points=self.border_facing_points,
            detail=f"closing on {track.target_zone} heading {track.heading} for {streak} frames",
        )

    def _erratic(self, track: TrackContext) -> Behaviour | None:
        if track.samples < self.erratic_min_samples or not track.is_moving:
            return None
        if track.heading_variance < self.erratic_threshold:
            return None
        return Behaviour(
            name=ERRATIC,
            points=self.erratic_points,
            detail=f"inconsistent heading (variance {track.heading_variance:.2f})",
        )

    def _night_movement(self, track: TrackContext) -> Behaviour | None:
        if not track.scene.is_night or not track.is_moving:
            return None
        return Behaviour(
            name=NIGHT_MOVEMENT,
            points=self.night_movement_points,
            detail=f"moving at {track.speed_label} during the night window",
        )

    def _running(self, track, label: str = "PERSON") -> Behaviour | None:
        """Requires calibration - see `running_speed_mps`."""
        threshold = self.running_speed_mps if is_person(label) else self.vehicle_speed_mps
        if track.speed_mps is None or track.speed_mps < threshold:
            return None
        return Behaviour(
            name=RUNNING,
            points=self.running_points,
            detail=f"{label.lower()} moving at {track.speed_label}, above the "
                   f"{threshold:g} {track.unit}/s threshold",
        )

    def _camera_tampering(self, track: TrackContext) -> Behaviour | None:
        """Someone close enough to touch, cover or turn the camera."""
        if track.height_ratio < self.tamper_height_ratio:
            return None
        detail = f"person fills {track.height_ratio:.0%} of frame height"
        if track.bottom_clipped:
            detail += ", feet out of shot"
        return Behaviour(name=CAMERA_TAMPERING, points=self.tamper_points, detail=detail)

    def _approaching_camera(self, track: TrackContext) -> Behaviour | None:
        """Growing quickly in frame: walking straight at the camera."""
        if track.height_ratio < self.camera_approach_min_ratio:
            return None
        if track.camera_growth < self.camera_approach_growth:
            return None
        return Behaviour(
            name=APPROACHING_CAMERA,
            points=self.camera_approach_points,
            detail=(
                f"apparent size growing {track.camera_growth:+.0%}/s "
                f"(now {track.height_ratio:.0%} of frame)"
            ),
        )

    def from_registry(self, check) -> list[Behaviour]:
        """Turn a registry comparison into behaviours the risk engine can score."""
        if check is None:
            return []
        if check.flagged:
            status = check.record.status if check.record else "FLAGGED"
            return [
                Behaviour(
                    name=WATCHLISTED,
                    points=self.watchlisted_points,
                    detail=f"{check.plate} is {status.lower()} in the registry",
                )
            ]
        if check.mismatch:
            return [
                Behaviour(
                    name=PLATE_MISMATCH,
                    points=self.plate_mismatch_points,
                    detail=check.detail,
                )
            ]
        if not check.known:
            return [
                Behaviour(
                    name=UNREGISTERED,
                    points=self.unregistered_points,
                    detail=check.detail,
                )
            ]
        return []

    def from_identity(self, match) -> list[Behaviour]:
        """Turn a roster match into behaviours the risk engine can score."""
        if match is None:
            return []
        if match.recognised and match.person is not None:
            label = match.person.label or match.person.person_id
            role = f", {match.person.role}" if match.person.role else ""
            return [
                Behaviour(
                    name=AUTHORISED,
                    points=self.authorised_points,
                    detail=f"recognised as {label}{role} ({match.similarity:.2f})",
                )
            ]
        if self.unrecognised_points:
            return [
                Behaviour(
                    name=UNRECOGNISED,
                    points=self.unrecognised_points,
                    detail="face not on the roster for this post",
                )
            ]
        return []

    def forget(self, track_id: int) -> None:
        self._approach_streak.pop(track_id, None)

    def reset(self) -> None:
        self._approach_streak.clear()
