"""SQLite event store.

The JSONL log is an append-only audit trail; this is the queryable one the
dashboard and API read from. SQLite is deliberate for now: it needs no server,
ships with Python, and the schema below is written so a move to PostgreSQL is
a connection-string change plus a dialect pass, not a redesign.

Every row also keeps the complete event JSON in `payload`, so adding a field
to the event model never loses data already written.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ..alerts.events import IntrusionEvent
from .integrity import GENESIS, ChainReport, link, verify

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY,
    timestamp     TEXT    NOT NULL,
    camera_id     TEXT    NOT NULL,
    event_type    TEXT    NOT NULL,
    frame_index   INTEGER,
    object_label  TEXT,
    confidence    REAL,
    track_id      INTEGER,
    track_label   TEXT,
    zone_name     TEXT,
    zone_kind     TEXT,
    risk_score    INTEGER NOT NULL,
    severity      TEXT    NOT NULL,
    reason        TEXT,
    behaviours    TEXT,
    evidence_path TEXT,
    clip_path     TEXT,
    plate         TEXT,
    payload       TEXT    NOT NULL,
    -- Chain of custody. Each event is hashed together with the hash of the one
    -- before it, so editing or deleting any past event breaks the link and the
    -- break points at the row. See app/store/integrity.py.
    prev_hash     TEXT,
    hash          TEXT,
    seq           INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_camera    ON events (camera_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_severity  ON events (severity, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_plate     ON events (plate);
"""


@dataclass
class EventStore:
    """Thread-safe SQLite store. One connection, guarded by a lock.

    A lock rather than a connection pool because the write rate here is one row
    per alert, not per frame - contention is never the bottleneck.
    """

    path: Path = Path("data/sentinelx.db")
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(SCHEMA)
            self._migrate()
            self._connection.commit()

    def _migrate(self) -> None:
        """Add the chain columns to a database written before they existed.

        Existing rows keep NULL hashes rather than being back-filled. A hash
        computed today over a row written last month would prove nothing about
        last month, and a chain that quietly claims to cover events it never
        saw is worse than one that admits where it starts.
        """
        existing = {row[1] for row in self._connection.execute("PRAGMA table_info(events)")}
        for column, kind in (("prev_hash", "TEXT"), ("hash", "TEXT"), ("seq", "INTEGER")):
            if column not in existing:
                self._connection.execute(f"ALTER TABLE events ADD COLUMN {column} {kind}")

    # ------------------------------------------------------ chain of custody

    def _head(self) -> tuple[str, int]:
        """The newest link in the chain, and the sequence number after it."""
        row = self._connection.execute(
            "SELECT hash, seq FROM events WHERE hash IS NOT NULL "
            "ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return GENESIS, 0
        return row["hash"], int(row["seq"] or 0) + 1

    def head(self) -> str:
        """The current head hash - the single value that fixes the whole log.

        Publishing this somewhere the holder of this database does not control
        is what turns a tamper-*evident* log into a chain of custody. Read it
        out at shift handover, send it to the district server, anchor it on a
        public chain: any of those work, and none of them happen here.
        """
        with self._lock:
            return self._head()[0]

    def verify_chain(self) -> ChainReport:
        """Recompute every link and report where, if anywhere, it breaks."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT event_id, timestamp, payload, prev_hash, hash FROM events "
                "ORDER BY seq ASC, timestamp ASC"
            ).fetchall()
        return verify(rows)

    # ------------------------------------------------------------- writing

    def write(self, event: IntrusionEvent) -> None:
        payload = event.to_dict()
        obj = payload.get("object") or {}
        zone = payload.get("zone") or {}
        row = (
            payload["event_id"],
            payload["timestamp"],
            payload["camera_id"],
            payload["event_type"],
            payload["frame_index"],
            obj.get("label"),
            obj.get("confidence"),
            obj.get("track_id"),
            obj.get("track_label"),
            zone.get("name"),
            zone.get("kind"),
            payload["risk"]["score"],
            payload["risk"]["severity"],
            payload["risk"]["reason"],
            json.dumps([b["name"] for b in payload.get("behaviours", [])]),
            payload.get("evidence_path"),
            payload.get("clip_path"),
            (payload.get("plate") or {}).get("text"),
            json.dumps(payload, ensure_ascii=False),
        )
        with self._lock:
            # The hash covers the JSON exactly as it is about to be stored, so
            # verification later reads the same bytes an auditor would.
            previous, seq = self._head()
            stored_json = row[-1]
            digest = link(previous, payload["event_id"], payload["timestamp"], stored_json)
            self._connection.execute(
                """INSERT OR REPLACE INTO events (
                    event_id, timestamp, camera_id, event_type, frame_index,
                    object_label, confidence, track_id, track_label,
                    zone_name, zone_kind, risk_score, severity, reason,
                    behaviours, evidence_path, clip_path, plate, payload,
                    prev_hash, hash, seq
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                row + (previous, digest, seq),
            )
            self._connection.commit()

    # ------------------------------------------------------------- reading

    def recent(
        self,
        limit: int = 50,
        camera_id: str | None = None,
        severity: str | None = None,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Most recent events first, optionally filtered."""
        clauses: list[str] = []
        params: list[Any] = []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if severity:
            clauses.append("severity = ?")
            params.append(severity.upper())
        if since:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat(timespec="seconds"))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 500)))
        with self._lock:
            rows = self._connection.execute(
                f"SELECT payload FROM events {where} ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                params,
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def by_plate(self, plate: str, limit: int = 50) -> list[dict[str, Any]]:
        """Every event involving one number plate - the movement history of a vehicle."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM events WHERE plate = ? "
                "ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                (plate.upper(), max(1, min(limit, 500))),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def plates_seen(self) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT plate, COUNT(*) AS n FROM events "
                "WHERE plate IS NOT NULL GROUP BY plate ORDER BY n DESC"
            ).fetchall()
        return {r["plate"]: int(r["n"]) for r in rows}

    def get(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def summary(self) -> dict[str, Any]:
        """Counts the dashboard header needs, in one round trip each."""
        with self._lock:
            total = self._connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            by_severity = self._connection.execute(
                "SELECT severity, COUNT(*) AS n FROM events GROUP BY severity"
            ).fetchall()
            by_camera = self._connection.execute(
                "SELECT camera_id, COUNT(*) AS n FROM events GROUP BY camera_id"
            ).fetchall()
            latest = self._connection.execute(
                "SELECT timestamp FROM events ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return {
            "total": int(total),
            "by_severity": {r["severity"]: int(r["n"]) for r in by_severity},
            "by_camera": {r["camera_id"]: int(r["n"]) for r in by_camera},
            "latest": latest["timestamp"] if latest else None,
        }

    def behaviour_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute("SELECT behaviours FROM events").fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            for name in json.loads(row["behaviours"] or "[]"):
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def incidents(
        self,
        limit: int = 50,
        gap_seconds: float = 120.0,
        camera_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Group events into incidents, newest first.

        An operator does not want four rows saying the same person is still in
        the restricted zone; they want one row saying so, with the worst moment
        on it. Events are grouped when they concern the same track on the same
        camera and follow each other inside `gap_seconds` - or, for vehicles,
        when they share a plate, which links sightings across cameras.

        This is a read-side view. It changes nothing about what is detected or
        stored, so an incident can always be expanded back into its events.
        """
        events = self.recent(limit=500, camera_id=camera_id)
        events.sort(key=lambda e: e["timestamp"])

        groups: list[dict[str, Any]] = []
        index: dict[tuple, int] = {}

        for event in events:
            plate = (event.get("plate") or {}).get("text")
            obj = event.get("object") or {}
            track = obj.get("track_id")
            # A plate follows a vehicle between cameras; a track id does not.
            key = ("plate", plate) if plate else (event["camera_id"], track)

            when = datetime.fromisoformat(event["timestamp"])
            position = index.get(key)
            if position is not None:
                group = groups[position]
                if (when - datetime.fromisoformat(group["ended"])).total_seconds() <= gap_seconds:
                    self._extend(group, event, when)
                    continue

            groups.append(self._new_incident(event, when))
            index[key] = len(groups) - 1

        groups.sort(key=lambda g: g["ended"], reverse=True)
        return groups[: max(1, min(limit, 500))]

    @staticmethod
    def _new_incident(event: dict, when: datetime) -> dict[str, Any]:
        obj = event.get("object") or {}
        plate = (event.get("plate") or {}).get("text")
        return {
            "incident_id": event["event_id"],
            "started": event["timestamp"],
            "ended": event["timestamp"],
            "cameras": [event["camera_id"]],
            "object": obj.get("label"),
            "track_label": obj.get("track_label"),
            "plate": plate,
            "peak_score": event["risk"]["score"],
            "peak_severity": event["risk"]["severity"],
            "peak_event_id": event["event_id"],
            "events": 1,
            "event_ids": [event["event_id"]],
            "behaviours": [b["name"] for b in event.get("behaviours", [])],
            "zones": [event["zone"]["name"]] if event.get("zone") else [],
        }

    @staticmethod
    def _extend(group: dict[str, Any], event: dict, when: datetime) -> None:
        group["ended"] = event["timestamp"]
        group["events"] += 1
        group["event_ids"].append(event["event_id"])
        if event["camera_id"] not in group["cameras"]:
            group["cameras"].append(event["camera_id"])
        if event.get("zone") and event["zone"]["name"] not in group["zones"]:
            group["zones"].append(event["zone"]["name"])
        for behaviour in event.get("behaviours", []):
            if behaviour["name"] not in group["behaviours"]:
                group["behaviours"].append(behaviour["name"])
        if not group["plate"]:
            group["plate"] = (event.get("plate") or {}).get("text")
        # The incident is as serious as its worst moment.
        if event["risk"]["score"] > group["peak_score"]:
            group["peak_score"] = event["risk"]["score"]
            group["peak_severity"] = event["risk"]["severity"]
            group["peak_event_id"] = event["event_id"]

    def purge(self) -> None:
        """Drop every row. Used by tests and by `--reset`."""
        with self._lock:
            self._connection.execute("DELETE FROM events")
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, *exc: Iterable[Any]) -> None:
        self.close()
