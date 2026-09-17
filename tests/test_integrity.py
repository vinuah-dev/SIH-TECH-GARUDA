"""Proving the event log has not been altered.

An alert from a border post can end up as evidence, and what matters then is
not that it was recorded but that nobody edited it afterwards - and that this
can be *shown* rather than asserted by whoever holds the database.

Every event is hashed together with the hash of the one before it. The tests
that matter are the destructive ones: edit a row, delete a row, reorder the
log, and check the break is found and points at the right place. A chain that
cannot be shown to catch tampering is decoration.

The honest limit is tested too. Anyone holding the database can *append* their
own events and the chain stays intact - that is what separates this from a
distributed ledger, and pretending otherwise would be the exact overclaim this
project exists to avoid.
"""

import json
import sqlite3

import pytest

from app.alerts.events import IntrusionEvent
from app.store import EventStore
from app.store.integrity import GENESIS, link, verify


def event(store, event_id, score=70, camera="CAM-01"):
    """One real event through the store's own writing path."""
    from datetime import datetime

    from app.detection.base import Detection
    from app.risk.engine import RiskAssessment
    from app.zones.manager import Zone

    store.write(IntrusionEvent(
        event_id=event_id,
        timestamp=datetime(2026, 9, 3, 12, 0, int(event_id[-2:], 16) % 60),
        camera_id=camera,
        event_type="VIRTUAL FENCE INTRUSION",
        frame_index=10,
        detection=Detection("PERSON", 0.9, (10, 10, 60, 200), track_id=1),
        zone=Zone("RESTRICTED", "RESTRICTED", [(0, 0), (1, 0), (1, 1)], base_risk=60),
        risk=RiskAssessment(score=score, severity="HIGH", reason="test", factors=()),
        behaviours=(),
    ))


@pytest.fixture
def store(tmp_path):
    with EventStore(tmp_path / "chain.db") as ready:
        yield ready


def raw(store):
    """The stored rows, in write order, as an auditor would read them."""
    connection = sqlite3.connect(str(store.path))
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT event_id, timestamp, payload, prev_hash, hash FROM events "
        "ORDER BY seq ASC"
    ).fetchall()
    connection.close()
    return rows


# ------------------------------------------------------------------ the chain


def test_an_empty_log_is_intact_and_starts_at_genesis(store):
    report = store.verify_chain()
    assert report.intact
    assert report.events == 0
    assert report.head == GENESIS


def test_every_event_is_chained_to_the_one_before_it(store):
    for i in range(4):
        event(store, f"EV{i:010d}")

    rows = raw(store)
    assert len(rows) == 4
    assert rows[0]["prev_hash"] == GENESIS
    for earlier, later in zip(rows, rows[1:]):
        assert later["prev_hash"] == earlier["hash"], "the link is not carried forward"
    assert store.verify_chain().intact


def test_the_head_moves_with_every_event(store):
    first = store.head()
    event(store, "EV0000000001")
    second = store.head()
    event(store, "EV0000000002")

    assert first == GENESIS
    assert second != first
    assert store.head() != second
    assert store.head() == raw(store)[-1]["hash"]


def test_two_logs_of_the_same_events_agree(tmp_path):
    """A district office can only compare posts if the same events chain alike."""
    heads = []
    for name in ("a.db", "b.db"):
        with EventStore(tmp_path / name) as store:
            for i in range(3):
                event(store, f"EV{i:010d}")
            heads.append(store.head())
    assert heads[0] == heads[1]


# -------------------------------------------------------------- tampering


def test_editing_an_event_is_caught_and_located(store):
    """The whole point. Change a risk score in the database and it shows."""
    for i in range(4):
        event(store, f"EV{i:010d}")
    assert store.verify_chain().intact

    connection = sqlite3.connect(str(store.path))
    row = connection.execute(
        "SELECT payload FROM events WHERE event_id = 'EV0000000002'").fetchone()
    payload = json.loads(row[0])
    payload["risk"]["score"] = 5                       # downgrade the incident
    connection.execute("UPDATE events SET payload = ? WHERE event_id = 'EV0000000002'",
                       (json.dumps(payload, ensure_ascii=False),))
    connection.commit()
    connection.close()

    report = store.verify_chain()
    assert not report.intact
    assert report.breaks[0].event_id == "EV0000000002", "the break must name the row"
    assert "changed" in report.breaks[0].reason


def test_deleting_an_event_is_caught(store):
    """Removing an inconvenient alert leaves a hole the next link exposes."""
    for i in range(4):
        event(store, f"EV{i:010d}")

    connection = sqlite3.connect(str(store.path))
    connection.execute("DELETE FROM events WHERE event_id = 'EV0000000001'")
    connection.commit()
    connection.close()

    report = store.verify_chain()
    assert not report.intact
    assert report.events == 3
    # The event after the hole is where the chain stops agreeing.
    assert report.breaks[0].event_id == "EV0000000002"


def test_reordering_the_log_is_caught(store):
    for i in range(3):
        event(store, f"EV{i:010d}")

    connection = sqlite3.connect(str(store.path))
    connection.execute("UPDATE events SET seq = 99 WHERE event_id = 'EV0000000000'")
    connection.commit()
    connection.close()

    assert not store.verify_chain().intact


def test_a_forged_hash_does_not_repair_an_edit(store):
    """Rewriting the row's own hash only moves the break to the next row.

    This is the property that makes the chain worth having: fixing one link
    requires rewriting every link after it.
    """
    for i in range(4):
        event(store, f"EV{i:010d}")

    connection = sqlite3.connect(str(store.path))
    connection.row_factory = sqlite3.Row
    target = connection.execute(
        "SELECT * FROM events WHERE event_id = 'EV0000000001'").fetchone()
    payload = json.loads(target["payload"])
    payload["risk"]["score"] = 1
    stored = json.dumps(payload, ensure_ascii=False)
    connection.execute(
        "UPDATE events SET payload = ?, hash = ? WHERE event_id = 'EV0000000001'",
        (stored, link(target["prev_hash"], target["event_id"], target["timestamp"], stored)),
    )
    connection.commit()
    connection.close()

    report = store.verify_chain()
    assert not report.intact
    assert report.breaks[0].event_id == "EV0000000002", (
        "the break should surface at the next row, whose prev_hash no longer matches"
    )


def test_every_break_is_reported_not_only_the_first(store):
    """One edited row and one deleted row are different findings."""
    for i in range(6):
        event(store, f"EV{i:010d}")

    connection = sqlite3.connect(str(store.path))
    connection.execute("UPDATE events SET payload = '{}' WHERE event_id = 'EV0000000001'")
    connection.execute("DELETE FROM events WHERE event_id = 'EV0000000004'")
    connection.commit()
    connection.close()

    report = store.verify_chain()
    assert len(report.breaks) >= 2


# ------------------------------------------------------- the honest limit


def test_appending_is_still_possible_and_the_chain_says_nothing(tmp_path):
    """This is tamper-evident, not tamper-proof, and the test says so.

    Anyone holding the database can extend the chain with their own events.
    What they cannot do is quietly change what is already in it. Publishing the
    head somewhere they do not control is what closes this gap, and nothing
    here does that for you.
    """
    with EventStore(tmp_path / "c.db") as store:
        for i in range(3):
            event(store, f"EV{i:010d}")
        honest_head = store.head()

        event(store, "EVFFFFFFFF01")                   # an appended event

        assert store.verify_chain().intact, (
            "appending must not look like tampering - it is not"
        )
        assert store.head() != honest_head, (
            "but the head moves, which is what a published head would expose"
        )


# ------------------------------------------------------ older databases


def test_a_database_written_before_the_chain_still_opens(tmp_path):
    """Adding integrity must not make an existing post's history unreadable."""
    path = tmp_path / "old.db"
    connection = sqlite3.connect(str(path))
    connection.executescript("""
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
            camera_id TEXT NOT NULL, event_type TEXT NOT NULL,
            frame_index INTEGER, object_label TEXT, confidence REAL,
            track_id INTEGER, track_label TEXT, zone_name TEXT, zone_kind TEXT,
            risk_score INTEGER NOT NULL, severity TEXT NOT NULL, reason TEXT,
            behaviours TEXT, evidence_path TEXT, clip_path TEXT, plate TEXT,
            payload TEXT NOT NULL);
        INSERT INTO events VALUES ('OLD1','2026-01-01T00:00:00','CAM-01','X',1,
            'PERSON',0.9,1,'P001','Z','RESTRICTED',70,'HIGH','r','[]',
            NULL,NULL,NULL,'{}');
    """)
    connection.commit()
    connection.close()

    with EventStore(path) as store:
        assert store.count() == 1
        report = store.verify_chain()
        # The old row is reported as unverifiable rather than silently trusted.
        assert not report.intact
        assert "before the chain existed" in report.breaks[0].reason


def test_old_rows_are_not_back_filled_with_hashes_computed_today(tmp_path):
    """A hash computed now proves nothing about a row written last month."""
    path = tmp_path / "old.db"
    connection = sqlite3.connect(str(path))
    connection.executescript("""
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
            camera_id TEXT NOT NULL, event_type TEXT NOT NULL,
            frame_index INTEGER, object_label TEXT, confidence REAL,
            track_id INTEGER, track_label TEXT, zone_name TEXT, zone_kind TEXT,
            risk_score INTEGER NOT NULL, severity TEXT NOT NULL, reason TEXT,
            behaviours TEXT, evidence_path TEXT, clip_path TEXT, plate TEXT,
            payload TEXT NOT NULL);
        INSERT INTO events VALUES ('OLD1','2026-01-01T00:00:00','CAM-01','X',1,
            'PERSON',0.9,1,'P001','Z','RESTRICTED',70,'HIGH','r','[]',
            NULL,NULL,NULL,'{}');
    """)
    connection.commit()
    connection.close()

    with EventStore(path):
        pass
    connection = sqlite3.connect(str(path))
    stored = connection.execute("SELECT hash FROM events WHERE event_id='OLD1'").fetchone()
    connection.close()
    assert stored[0] is None, "an old row must not be given a hash it never had"


# --------------------------------------------------------------- the maths


def test_the_hash_depends_on_every_part_of_the_event():
    base = link("aa", "EV1", "2026-09-03T12:00:00", '{"score":70}')
    assert base != link("bb", "EV1", "2026-09-03T12:00:00", '{"score":70}')
    assert base != link("aa", "EV2", "2026-09-03T12:00:00", '{"score":70}')
    assert base != link("aa", "EV1", "2026-09-03T12:00:01", '{"score":70}')
    assert base != link("aa", "EV1", "2026-09-03T12:00:00", '{"score":71}')


def test_field_boundaries_cannot_be_slid():
    """Naive concatenation lets one field borrow from the next; this must not."""
    assert link("a", "bc", "d", "e") != link("ab", "c", "d", "e")


def test_verify_reads_rows_it_is_given(store):
    for i in range(3):
        event(store, f"EV{i:010d}")
    assert verify(raw(store)).intact
