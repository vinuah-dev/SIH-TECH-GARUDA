"""SQLite event store: writing, querying and surviving a restart."""

from datetime import datetime

import pytest

from app.alerts.events import IntrusionEvent
from app.behaviour.engine import Behaviour
from app.detection.base import Detection
from app.risk.engine import RiskAssessment, RiskFactor
from app.store import EventStore
from app.zones.manager import Zone

ZONE = Zone("RESTRICTED", "RESTRICTED", [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0)], base_risk=60)


def make_event(event_id="A1", camera="CAM-01", score=88, severity="HIGH", hour=12,
               behaviours=("BORDER-FACING MOVEMENT",)):
    return IntrusionEvent(
        event_id=event_id,
        timestamp=datetime(2026, 8, 25, hour, 0, 0),
        camera_id=camera,
        event_type="VIRTUAL FENCE INTRUSION",
        detection=Detection("PERSON", 0.9, (10, 10, 60, 200), track_id=1),
        zone=ZONE,
        risk=RiskAssessment(
            score=score,
            severity=severity,
            reason="test",
            factors=[RiskFactor("RESTRICTED ZONE", score, "test")],
        ),
        frame_index=42,
        evidence_path="data/evidence/x.jpg",
        behaviours=[Behaviour(name, 8, "d") for name in behaviours],
    )


@pytest.fixture
def store(tmp_path):
    with EventStore(tmp_path / "test.db") as s:
        yield s


# ------------------------------------------------------------------ writing


def test_starts_empty(store):
    assert store.count() == 0
    assert store.recent() == []


def test_written_event_is_readable(store):
    store.write(make_event())
    assert store.count() == 1
    assert store.recent()[0]["event_id"] == "A1"


def test_full_event_survives_the_round_trip(store):
    event = make_event()
    store.write(event)
    assert store.recent()[0] == event.to_dict()


def test_writing_the_same_id_twice_does_not_duplicate(store):
    store.write(make_event(event_id="A1"))
    store.write(make_event(event_id="A1", score=95, severity="CRITICAL"))
    assert store.count() == 1
    assert store.recent()[0]["risk"]["score"] == 95


def test_creates_its_parent_directory(tmp_path):
    store = EventStore(tmp_path / "nested" / "deeper" / "sentinelx.db")
    store.write(make_event())
    assert store.count() == 1
    store.close()


# ------------------------------------------------------------------ reading


def test_recent_returns_newest_first(store):
    store.write(make_event(event_id="old", hour=9))
    store.write(make_event(event_id="new", hour=18))
    assert [e["event_id"] for e in store.recent()] == ["new", "old"]


def test_recent_respects_the_limit(store):
    for i in range(10):
        store.write(make_event(event_id=f"E{i}", hour=i))
    assert len(store.recent(limit=3)) == 3


def test_limit_is_clamped_to_something_sane(store):
    store.write(make_event())
    assert len(store.recent(limit=10_000)) == 1
    assert len(store.recent(limit=0)) == 1


def test_filter_by_camera(store):
    store.write(make_event(event_id="A", camera="CAM-01"))
    store.write(make_event(event_id="B", camera="CAM-02"))
    assert [e["event_id"] for e in store.recent(camera_id="CAM-02")] == ["B"]


def test_filter_by_severity_is_case_insensitive(store):
    store.write(make_event(event_id="H", severity="HIGH"))
    store.write(make_event(event_id="C", severity="CRITICAL"))
    assert [e["event_id"] for e in store.recent(severity="critical")] == ["C"]


def test_filter_since_a_timestamp(store):
    store.write(make_event(event_id="early", hour=8))
    store.write(make_event(event_id="late", hour=20))
    found = store.recent(since=datetime(2026, 8, 25, 12, 0))
    assert [e["event_id"] for e in found] == ["late"]


def test_get_by_id(store):
    store.write(make_event(event_id="FIND-ME"))
    assert store.get("FIND-ME")["event_id"] == "FIND-ME"
    assert store.get("NOPE") is None


# ---------------------------------------------------------------- aggregates


def test_summary_counts_by_severity_and_camera(store):
    store.write(make_event(event_id="A", camera="CAM-01", severity="HIGH"))
    store.write(make_event(event_id="B", camera="CAM-01", severity="CRITICAL"))
    store.write(make_event(event_id="C", camera="CAM-02", severity="HIGH"))

    summary = store.summary()
    assert summary["total"] == 3
    assert summary["by_severity"] == {"HIGH": 2, "CRITICAL": 1}
    assert summary["by_camera"] == {"CAM-01": 2, "CAM-02": 1}
    assert summary["latest"] is not None


def test_summary_of_an_empty_store(store):
    assert store.summary() == {"total": 0, "by_severity": {}, "by_camera": {}, "latest": None}


def test_behaviour_counts_are_ranked(store):
    store.write(make_event(event_id="A", behaviours=("LOITERING", "NIGHT MOVEMENT")))
    store.write(make_event(event_id="B", behaviours=("LOITERING",)))
    counts = store.behaviour_counts()
    assert counts == {"LOITERING": 2, "NIGHT MOVEMENT": 1}
    assert list(counts) == ["LOITERING", "NIGHT MOVEMENT"]


# ------------------------------------------------------------- housekeeping


def test_data_survives_reopening(tmp_path):
    path = tmp_path / "persist.db"
    first = EventStore(path)
    first.write(make_event(event_id="KEEP"))
    first.close()

    second = EventStore(path)
    assert second.get("KEEP") is not None
    second.close()


def test_purge_empties_the_store(store):
    store.write(make_event())
    store.purge()
    assert store.count() == 0


def test_concurrent_writers_do_not_lose_events(tmp_path):
    """The multi-camera runner will write from one thread per camera."""
    import threading

    store = EventStore(tmp_path / "threads.db")
    errors: list[Exception] = []

    def writer(prefix: str) -> None:
        try:
            for i in range(25):
                store.write(make_event(event_id=f"{prefix}-{i}", camera=prefix))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(f"CAM-{n}",)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert store.count() == 100
    store.close()


# --------------------------------------------------------------- incidents


def make_vehicle_event(event_id, camera, track, hour, minute, score, severity, plate=None,
                       behaviours=()):
    from app.behaviour.engine import Behaviour

    event = make_event(event_id=event_id, camera=camera, score=score, severity=severity,
                       hour=hour, behaviours=behaviours)
    return IntrusionEvent(
        event_id=event.event_id,
        timestamp=datetime(2026, 8, 25, hour, minute),
        camera_id=camera,
        event_type=event.event_type,
        detection=Detection("PERSON", 0.9, (10, 10, 60, 200), track_id=track),
        zone=ZONE,
        risk=event.risk,
        frame_index=1,
        behaviours=event.behaviours,
        plate={"text": plate, "display": plate, "state": None, "confidence": 0.9,
               "reads": 3, "bbox": [], "raw": plate} if plate else None,
    )


def test_repeated_alerts_about_one_person_become_one_incident(store):
    """Four rows saying the same thing is four chances to stop reading them."""
    for i, (score, sev) in enumerate([(70, "HIGH"), (80, "HIGH"), (95, "CRITICAL")]):
        store.write(make_vehicle_event(f"A{i}", "CAM-01", 1, 2, i, score, sev))

    incidents = store.incidents()
    assert len(incidents) == 1
    assert incidents[0]["events"] == 3
    assert incidents[0]["peak_score"] == 95
    assert incidents[0]["peak_severity"] == "CRITICAL"


def test_an_incident_keeps_every_behaviour_seen_during_it():
    """The union, because a threat that escalates shows different signs over time."""
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        with EventStore(_Path(tmp) / "b.db") as store:
            store.write(make_vehicle_event("A", "CAM-01", 1, 2, 0, 70, "HIGH",
                                           behaviours=("BORDER-FACING MOVEMENT",)))
            store.write(make_vehicle_event("B", "CAM-01", 1, 2, 1, 95, "CRITICAL",
                                           behaviours=("LOITERING",)))
            found = store.incidents()[0]
            assert set(found["behaviours"]) == {"BORDER-FACING MOVEMENT", "LOITERING"}


def test_a_long_gap_starts_a_new_incident(store):
    store.write(make_vehicle_event("A", "CAM-01", 1, 2, 0, 70, "HIGH"))
    store.write(make_vehicle_event("B", "CAM-01", 1, 5, 0, 70, "HIGH"))
    assert len(store.incidents(gap_seconds=60)) == 2


def test_different_tracks_are_different_incidents(store):
    store.write(make_vehicle_event("A", "CAM-01", 1, 2, 0, 70, "HIGH"))
    store.write(make_vehicle_event("B", "CAM-01", 2, 2, 0, 70, "HIGH"))
    assert len(store.incidents()) == 2


def test_a_plate_links_one_vehicle_across_cameras(store):
    """A track id does not survive between cameras; a plate does."""
    store.write(make_vehicle_event("A", "CAM-01", 1, 2, 0, 75, "HIGH", plate="MH12AB1234"))
    store.write(make_vehicle_event("B", "CAM-02", 9, 2, 1, 92, "CRITICAL", plate="MH12AB1234"))

    incidents = store.incidents()
    assert len(incidents) == 1
    assert set(incidents[0]["cameras"]) == {"CAM-01", "CAM-02"}
    assert incidents[0]["plate"] == "MH12AB1234"
    assert incidents[0]["peak_score"] == 92


def test_an_incident_can_be_expanded_back_into_its_events(store):
    """Grouping is a view; nothing is lost."""
    store.write(make_vehicle_event("A", "CAM-01", 1, 2, 0, 70, "HIGH"))
    store.write(make_vehicle_event("B", "CAM-01", 1, 2, 1, 95, "CRITICAL"))

    incident = store.incidents()[0]
    assert set(incident["event_ids"]) == {"A", "B"}
    for event_id in incident["event_ids"]:
        assert store.get(event_id) is not None


def test_incidents_of_an_empty_store(store):
    assert store.incidents() == []
