"""Scene-level context: facts true of the whole frame, not of one person.

Per-track movement context lives in `app/context/engine.py`; this module only
covers camera identity and time of day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SceneContext:
    camera_id: str
    timestamp: datetime
    is_night: bool

    @classmethod
    def build(
        cls,
        camera_id: str,
        timestamp: datetime | None = None,
        night_start: int = 18,
        night_end: int = 6,
        force_night: bool | None = None,
    ) -> "SceneContext":
        timestamp = timestamp or datetime.now()
        if force_night is None:
            hour = timestamp.hour
            # Night window wraps past midnight.
            is_night = hour >= night_start or hour < night_end
        else:
            is_night = force_night
        return cls(camera_id=camera_id, timestamp=timestamp, is_night=is_night)

    @property
    def time_of_day(self) -> str:
        return "NIGHT" if self.is_night else "DAY"
