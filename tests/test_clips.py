"""Evidence clips: the seconds either side of an alert.

A still frame proves someone was inside the fence; it does not show how they
got there or what they did next. The pre-roll is the part that cannot be
recovered any other way - by the time an alert exists, the approach is already
in the past - so these tests focus on it being genuinely captured.
"""

import cv2
import numpy as np
import pytest

from app.alerts.clips import ClipRecorder

WHEN = __import__("datetime").datetime(2026, 8, 25, 23, 30)


def frame(value=60, width=320, height=180):
    return np.full((height, width, 3), value, np.uint8)


def read_clip(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(image)
    capture.release()
    return frames


@pytest.fixture
def recorder(tmp_path):
    return ClipRecorder(tmp_path, pre_seconds=1.0, post_seconds=1.0, fps=10.0)


# ------------------------------------------------------------------ buffer


def test_the_buffer_holds_only_the_pre_roll(recorder):
    for i in range(50):
        recorder.observe(frame(), i * 0.1)
    # 1 second at 10 fps.
    assert recorder.buffered_frames == 10


def test_nothing_is_buffered_when_disabled(tmp_path):
    recorder = ClipRecorder(tmp_path, enabled=False)
    recorder.observe(frame(), 0.0)
    assert recorder.buffered_frames == 0
    assert recorder.start("E1", "CAM-01", WHEN, 0.0) is None


def test_a_disabled_recorder_creates_no_directory(tmp_path):
    ClipRecorder(tmp_path / "nothing", enabled=False)
    assert not (tmp_path / "nothing").exists()


def test_frames_are_shrunk_to_save_memory(tmp_path):
    recorder = ClipRecorder(tmp_path, max_width=160)
    recorder.observe(frame(width=640, height=360), 0.0)
    for _, stored in recorder._buffer:
        assert stored.shape[1] == 160


def test_small_frames_are_left_alone(tmp_path):
    recorder = ClipRecorder(tmp_path, max_width=640)
    recorder.observe(frame(width=320, height=180), 0.0)
    assert recorder._buffer[0][1].shape[1] == 320


# --------------------------------------------------------------- recording


def test_a_clip_contains_the_run_up_to_the_alert(recorder, tmp_path):
    """The pre-roll is the whole point: it cannot be captured any other way."""
    for i in range(10):
        recorder.observe(frame(value=10 + i), i * 0.1)  # before the alert

    path = recorder.start("E1", "CAM-01", WHEN, 1.0)
    for i in range(15):  # runs past the 1 s post-roll so the clip is written
        recorder.observe(frame(value=200), 1.0 + i * 0.1)  # after the alert

    frames = read_clip(path)
    assert len(frames) > 10
    # The first frames are the darker run-up, the last are the bright aftermath.
    assert frames[0].mean() < frames[-1].mean()


def test_a_clip_is_written_only_once_the_post_roll_completes(recorder, tmp_path):
    recorder.observe(frame(), 0.0)
    path = recorder.start("E1", "CAM-01", WHEN, 0.0)

    recorder.observe(frame(), 0.5)
    assert recorder.pending_count == 1
    assert not path.exists()

    recorder.observe(frame(), 1.5)  # post-roll satisfied
    assert recorder.pending_count == 0
    assert path.exists()
    assert recorder.written == 1


def test_the_clip_filename_identifies_the_event(recorder):
    recorder.observe(frame(), 0.0)
    path = recorder.start("ABC123", "CAM-07", WHEN, 0.0)
    assert "CAM-07" in path.name
    assert "ABC123" in path.name
    assert path.suffix == ".mp4"


def test_two_alerts_produce_two_clips(recorder, tmp_path):
    recorder.observe(frame(), 0.0)
    first = recorder.start("E1", "CAM-01", WHEN, 0.0)
    second = recorder.start("E2", "CAM-01", WHEN, 0.2)
    assert recorder.pending_count == 2

    for i in range(20):
        recorder.observe(frame(), 0.3 + i * 0.1)

    assert first.exists() and second.exists()
    assert first != second
    assert recorder.written == 2


def test_flush_writes_a_clip_cut_short_by_shutdown(recorder):
    recorder.observe(frame(), 0.0)
    path = recorder.start("E1", "CAM-01", WHEN, 0.0)
    recorder.observe(frame(), 0.1)
    assert not path.exists()

    recorder.flush()
    assert path.exists()
    assert recorder.pending_count == 0


def test_flush_is_safe_with_nothing_pending(recorder):
    recorder.flush()
    assert recorder.written == 0


def test_a_resolution_change_mid_stream_does_not_corrupt_the_clip(recorder):
    """Some cameras renegotiate resolution after a reconnect."""
    for i in range(5):
        recorder.observe(frame(width=320, height=180), i * 0.1)
    path = recorder.start("E1", "CAM-01", WHEN, 0.5)
    for i in range(15):
        recorder.observe(frame(width=240, height=135), 0.6 + i * 0.1)

    frames = read_clip(path)
    assert frames, "clip should still be readable"
    assert len({f.shape for f in frames}) == 1


def test_an_alert_with_no_buffered_frames_still_works(recorder):
    """A camera that alerts on its very first frame must not crash."""
    path = recorder.start("E1", "CAM-01", WHEN, 0.0)
    for i in range(15):
        recorder.observe(frame(), i * 0.1)
    assert path.exists()


# ------------------------------------------------------------- integration


def test_the_pipeline_records_a_clip_alongside_the_snapshot(tmp_path):
    from app.config import SurveillanceConfig
    from app.pipeline import SurveillancePipeline

    config = SurveillanceConfig(
        source="synthetic",
        detector="sim",
        synthetic_frames=120,
        force_night=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "events",
        evidence_dir=tmp_path / "evidence",
        clip_pre_seconds=1.0,
        clip_post_seconds=1.0,
        color=False,
    )
    stats = SurveillancePipeline(config).run()
    assert stats.alerts == 1

    clips = list((tmp_path / "evidence").glob("*.mp4"))
    snapshots = list((tmp_path / "evidence").glob("*.jpg"))
    assert len(clips) == 1
    assert len(snapshots) == 1
    # Both artefacts belong to the same event.
    assert clips[0].stem == snapshots[0].stem
    assert read_clip(clips[0])


def test_the_event_record_points_at_the_clip(tmp_path):
    import json

    from app.config import SurveillanceConfig
    from app.pipeline import SurveillancePipeline

    config = SurveillanceConfig(
        source="synthetic",
        detector="sim",
        synthetic_frames=120,
        force_night=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "events",
        evidence_dir=tmp_path / "evidence",
        clip_pre_seconds=1.0,
        clip_post_seconds=1.0,
        color=False,
    )
    SurveillancePipeline(config).run()

    log = next((tmp_path / "events").glob("*.jsonl"))
    record = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert record["clip_path"].endswith(".mp4")
    assert record["evidence_path"].endswith(".jpg")


def test_clips_can_be_disabled(tmp_path):
    from app.config import SurveillanceConfig
    from app.pipeline import SurveillancePipeline

    config = SurveillanceConfig(
        source="synthetic",
        detector="sim",
        synthetic_frames=120,
        force_night=True,
        zones_path="config/zones.json",
        events_dir=tmp_path / "events",
        evidence_dir=tmp_path / "evidence",
        save_clips=False,
        color=False,
    )
    SurveillancePipeline(config).run()
    assert list((tmp_path / "evidence").glob("*.mp4")) == []
    assert list((tmp_path / "evidence").glob("*.jpg"))
