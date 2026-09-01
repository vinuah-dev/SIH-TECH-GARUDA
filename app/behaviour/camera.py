"""Camera integrity: treating the link itself as something worth watching.

A camera that stops answering is not only an operations problem. Cutting the
cable, unplugging the NVR, or knocking the housing off its mount all look
identical from here - the feed simply stops - and all three are things an
intruder does *before* crossing a fence, precisely so nothing records it.

The judgement this module encodes:

  * a blip that heals within `outage_grace` is treated as network noise
  * an outage that lasts longer is a security event
  * a camera that never comes back is the most serious event this system can
    raise, because from that moment on it is blind
  * repeated short drops are a pattern, not an accident - somebody is
    interfering with it

Everything here runs off link state transitions, so it needs no clock of its
own and works the same on a live camera as it does in a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..video.stream import CameraState
from .engine import Behaviour

LINK_LOST = "CAMERA LINK LOST"
CAMERA_OFFLINE = "CAMERA OFFLINE"
REPEATED_LINK_LOSS = "REPEATED LINK LOSS"


@dataclass
class CameraIntegrityMonitor:
    """Turns camera link transitions into scored security behaviours."""

    # A drop shorter than this is almost always the network, not a person.
    outage_grace: float = 10.0
    # More than `repeat_threshold` drops inside this window is interference.
    repeat_window: float = 300.0
    repeat_threshold: int = 3
    # Do not re-raise the same finding while it is still obviously true.
    cooldown: float = 60.0

    link_lost_points: int = 40
    offline_points: int = 55
    repeated_points: int = 15

    _down_since: float | None = field(default=None, init=False)
    _drops: list[float] = field(default_factory=list, init=False)
    _last_alert: dict[str, float] = field(default_factory=dict, init=False)

    def observe(self, state: CameraState, now: float) -> list[Behaviour]:
        """Feed one link state transition. Returns behaviours worth alerting on."""
        if state is CameraState.RECONNECTING:
            return self._on_drop(now)
        if state is CameraState.ONLINE:
            return self._on_recovery(now)
        if state is CameraState.OFFLINE:
            return self._on_offline(now)
        return []

    # ------------------------------------------------------------ transitions

    def _on_drop(self, now: float) -> list[Behaviour]:
        if self._down_since is None:
            self._down_since = now
            self._drops.append(now)
        # Nothing is raised yet: the outage might heal within the grace period.
        # A *pattern* of drops is reportable immediately, though.
        return self._repeated(now)

    def _on_recovery(self, now: float) -> list[Behaviour]:
        if self._down_since is None:
            return []
        outage = now - self._down_since
        self._down_since = None

        behaviours: list[Behaviour] = []
        if outage >= self.outage_grace and self._allowed(LINK_LOST, now):
            behaviours.append(
                Behaviour(
                    name=LINK_LOST,
                    points=self.link_lost_points,
                    detail=f"feed lost for {outage:.0f}s, then restored",
                )
            )
        return behaviours + self._repeated(now)

    def _on_offline(self, now: float) -> list[Behaviour]:
        if not self._allowed(CAMERA_OFFLINE, now):
            return []
        if self._down_since is not None:
            detail = f"no feed for {now - self._down_since:.0f}s and not recovering"
        else:
            detail = "camera did not come back"
        self._down_since = None
        return [
            Behaviour(name=CAMERA_OFFLINE, points=self.offline_points, detail=detail)
        ] + self._repeated(now)

    # --------------------------------------------------------------- helpers

    def _repeated(self, now: float) -> list[Behaviour]:
        """Several drops in a short window reads as deliberate interference."""
        self._drops = [t for t in self._drops if now - t <= self.repeat_window]
        if len(self._drops) < self.repeat_threshold:
            return []
        if not self._allowed(REPEATED_LINK_LOSS, now):
            return []
        window_minutes = self.repeat_window / 60
        return [
            Behaviour(
                name=REPEATED_LINK_LOSS,
                points=self.repeated_points,
                detail=f"{len(self._drops)} link drops in the last {window_minutes:.0f} min",
            )
        ]

    def _allowed(self, name: str, now: float) -> bool:
        last = self._last_alert.get(name)
        if last is not None and now - last < self.cooldown:
            return False
        self._last_alert[name] = now
        return True

    def reset(self) -> None:
        self._down_since = None
        self._drops.clear()
        self._last_alert.clear()
