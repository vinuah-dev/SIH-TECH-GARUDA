"""End-to-end pipeline test.

Uses the simulated detector so the whole chain (ingest -> detect -> track ->
zone -> risk -> alert -> log -> evidence) runs without model weights.
"""

import json

import pytest

from app.config import SurveillanceConfig
from app.pipeline import SurveillancePipeline


def make_config(tmp_path, **overrides):
    defaults = dict(
        save_clips=False,  # clips are exercised in test_clips.py
        source="synthetic",
        detector="sim",
        synthetic_frames=40,
        force_night=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "events",
        evidence_dir=tmp_path / "evidence",
        color=False,
    )
    defaults.update(overrides)
    return SurveillanceConfig(**defaults)


def test_walker_crossing_the_fence_raises_exactly_one_alert(tmp_path):
    stats = SurveillancePipeline(make_config(tmp_path)).run()
    assert stats.frames == 40
    assert stats.detections == 40
    assert stats.alerts == 1


def test_alert_is_written_to_the_event_log(tmp_path):
    SurveillancePipeline(make_config(tmp_path)).run()
    logs = list((tmp_path / "events").glob("*.jsonl"))
    assert len(logs) == 1

    records = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["event_type"] == "VIRTUAL FENCE INTRUSION"
    assert record["zone"]["name"] == "RESTRICTED"
    assert record["object"]["label"] == "PERSON"
    # Restricted (60) + confidence (10) + night (10) + approaching (8).
    assert record["risk"]["score"] == 88
    assert record["risk"]["severity"] == "HIGH"
    assert sum(f["points"] for f in record["risk"]["factors"]) == record["risk"]["score"]

    context = record["context"]
    assert context["heading"] == "E"
    assert context["approaching"] is True
    assert context["target_zone"] == "RESTRICTED"
    assert context["speed"] > 0


def test_evidence_snapshot_is_saved(tmp_path):
    SurveillancePipeline(make_config(tmp_path)).run()
    snapshots = list((tmp_path / "evidence").glob("*.jpg"))
    assert len(snapshots) == 1
    assert snapshots[0].stat().st_size > 0


def test_evidence_can_be_disabled(tmp_path):
    config = make_config(tmp_path, save_evidence=False, save_clips=False)
    SurveillancePipeline(config).run()
    assert not (tmp_path / "evidence").exists()


def test_no_alert_when_no_zone_kind_is_armed(tmp_path):
    config = make_config(tmp_path, alert_kinds=frozenset({"PATROL"}))
    stats = SurveillancePipeline(config).run()
    assert stats.frames == 40
    assert stats.alerts == 0


def test_daytime_run_scores_lower_than_night(tmp_path):
    SurveillancePipeline(make_config(tmp_path / "night", force_night=True)).run()
    SurveillancePipeline(make_config(tmp_path / "day", force_night=False)).run()

    def score(root):
        log = next((root / "events").glob("*.jsonl"))
        return json.loads(log.read_text(encoding="utf-8").splitlines()[0])["risk"]["score"]

    assert score(tmp_path / "night") > score(tmp_path / "day")


def test_missing_source_is_reported_clearly(tmp_path):
    config = make_config(tmp_path, source="data/samples/does-not-exist.mp4")
    with pytest.raises(FileNotFoundError):
        SurveillancePipeline(config).run()


def test_invalid_detector_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="detector"):
        make_config(tmp_path, detector="magic")


def test_still_image_lowers_the_confirmation_requirement(tmp_path):
    """A single frame can never satisfy confirm_frames > 1."""
    import cv2
    import numpy as np

    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.zeros((540, 960, 3), dtype=np.uint8))

    config = make_config(tmp_path, source=str(still), confirm_frames=3)
    SurveillancePipeline(config).run()
    assert config.confirm_frames == 1


def test_loitering_escalates_to_a_second_critical_alert(tmp_path):
    """Entry alerts HIGH on approach; standing still escalates it to CRITICAL."""
    config = make_config(
        tmp_path, synthetic_frames=360, sim_walk_fraction=0.45, cooldown_seconds=30
    )
    stats = SurveillancePipeline(config).run()
    assert stats.alerts == 2

    log = next((tmp_path / "events").glob("*.jsonl"))
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    first, second = records

    assert first["risk"]["severity"] == "HIGH"
    entry_behaviours = {b["name"] for b in first["behaviours"]}
    assert "BORDER-FACING MOVEMENT" in entry_behaviours
    assert "NIGHT MOVEMENT" in entry_behaviours
    assert "LOITERING" not in entry_behaviours

    assert second["risk"]["severity"] == "CRITICAL"
    assert second["risk"]["score"] > first["risk"]["score"]
    loitering = next(b for b in second["behaviours"] if b["name"] == "LOITERING")
    assert loitering["points"] > 0
    # Standing still ends the approach signal - loitering has to carry the score.
    assert "BORDER-FACING MOVEMENT" not in {b["name"] for b in second["behaviours"]}
    assert second["context"]["dwell_seconds"] >= 10


def test_alerts_reach_the_sqlite_store(tmp_path):
    """The --db path: pipeline -> EventStore, queryable afterwards."""
    from app.store import EventStore

    db = tmp_path / "sentinelx.db"
    config = make_config(tmp_path, db_path=db)
    stats = SurveillancePipeline(config, store=EventStore(db)).run()

    with EventStore(db) as store:
        assert store.count() == stats.alerts == 1
        record = store.recent()[0]
        assert record["camera_id"] == "CAM-01"
        assert record["risk"]["severity"] == "HIGH"
        assert store.summary()["by_camera"] == {"CAM-01": 1}


def test_event_hooks_receive_every_alert(tmp_path):
    """Hooks are how the API server will observe alerts without coupling."""
    seen = []
    config = make_config(tmp_path, synthetic_frames=360, sim_walk_fraction=0.45)
    stats = SurveillancePipeline(config, event_hooks=[seen.append]).run()

    assert len(seen) == stats.alerts == 2
    assert [e.risk.severity for e in seen] == ["HIGH", "CRITICAL"]


def test_a_failing_hook_does_not_stop_the_feed(tmp_path):
    def explode(event):
        raise RuntimeError("hook is broken")

    seen = []
    config = make_config(tmp_path)
    stats = SurveillancePipeline(config, event_hooks=[explode, seen.append]).run()

    assert stats.alerts == 1
    assert len(seen) == 1


# ------------------------------------------- frame rate vs the alert policy


def slow_pipeline(tmp_path, fps, confirm_frames=3, frames=50):
    """A pipeline that believes it has been running at `fps`."""
    import time as _time

    config = make_config(tmp_path, confirm_frames=confirm_frames)
    pipeline = SurveillancePipeline(config)
    pipeline.stats.frames = frames
    pipeline.stats.started_at = _time.time() - frames / fps
    return pipeline


def test_a_slow_camera_warns_that_confirmation_takes_too_long(tmp_path, capsys):
    """Three frames is a fifth of a second on one camera and 2.3 s on eight.

    The policy is the same; the time it costs is not, and at the slow end a
    running person can cross a zone between the frames that would confirm them.
    """
    pipeline = slow_pipeline(tmp_path, fps=1.3)
    pipeline._check_confirmation_window()

    warning = capsys.readouterr().out
    assert "1.3 fps" in warning
    assert "2.3s" in warning


def test_a_healthy_frame_rate_says_nothing(tmp_path, capsys):
    pipeline = slow_pipeline(tmp_path, fps=13.0)
    pipeline._check_confirmation_window()
    assert capsys.readouterr().out == ""


def test_the_warning_is_raised_once_not_every_frame(tmp_path, capsys):
    pipeline = slow_pipeline(tmp_path, fps=1.3)
    for _ in range(5):
        pipeline._check_confirmation_window()
    assert capsys.readouterr().out.count("confirmation frames") == 1


def test_no_warning_before_there_is_enough_to_measure(tmp_path, capsys):
    """A handful of frames says nothing about sustained throughput."""
    pipeline = slow_pipeline(tmp_path, fps=1.3, frames=5)
    pipeline._check_confirmation_window()
    assert capsys.readouterr().out == ""


def test_fewer_confirmation_frames_shortens_the_window(tmp_path, capsys):
    """The advice the warning gives has to actually work."""
    pipeline = slow_pipeline(tmp_path, fps=1.3, confirm_frames=1)
    pipeline._check_confirmation_window()
    assert capsys.readouterr().out == ""


def test_the_warning_does_not_recommend_stride(tmp_path, capsys):
    """Stride was recommended here before it was measured, and it is wrong.

    Analysing every Nth frame lengthens the confirmation window rather than
    shortening it, so the advice a slow camera gets must not point that way.
    """
    slow_pipeline(tmp_path, fps=1.3)._check_confirmation_window()
    advice = capsys.readouterr().out

    assert "--confirm-frames" in advice
    assert "fewer cameras" in advice
    assert "Do NOT raise --stride" in advice
