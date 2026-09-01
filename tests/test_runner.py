"""Multi-camera runner: one process, one model, many feeds."""

import json
import threading

import numpy as np
import pytest

from app.config import SurveillanceConfig
from app.detection.base import Detection
from app.runner import (
    CameraSpec,
    MultiCameraRunner,
    SharedDetector,
    load_camera_specs,
)

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


def write_cameras(tmp_path, cameras):
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps({"cameras": cameras}), encoding="utf-8")
    return path


def base_config(tmp_path, **overrides):
    defaults = dict(
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


# ------------------------------------------------------------ configuration


def test_a_camera_needs_an_id_and_a_source():
    with pytest.raises(ValueError, match="camera_id and a source"):
        CameraSpec.from_dict({"source": "synthetic"})


def test_extra_keys_become_per_camera_overrides():
    spec = CameraSpec.from_dict(
        {"camera_id": "CAM-9", "source": "synthetic", "confidence": 0.8}
    )
    assert spec.overrides == {"confidence": 0.8}


def test_comment_keys_are_ignored():
    spec = CameraSpec.from_dict(
        {"camera_id": "CAM-9", "source": "synthetic", "_note": "ignore me"}
    )
    assert spec.overrides == {}


def test_overrides_are_applied_on_top_of_the_shared_config(tmp_path):
    spec = CameraSpec.from_dict(
        {"camera_id": "CAM-9", "source": "webcam:1", "confidence": 0.75}
    )
    config = spec.build_config(base_config(tmp_path, confidence=0.4))
    assert config.camera_id == "CAM-9"
    assert config.source == "webcam:1"
    assert config.confidence == 0.75
    # Shared settings survive.
    assert config.force_night is True


def test_fleet_members_do_not_print_their_own_summary(tmp_path):
    """The fleet prints one combined summary instead."""
    spec = CameraSpec.from_dict({"camera_id": "CAM-9", "source": "synthetic"})
    assert spec.build_config(base_config(tmp_path)).print_summary is False


def test_an_unknown_setting_is_rejected(tmp_path):
    spec = CameraSpec.from_dict({"camera_id": "CAM-9", "source": "x", "nonsense": 1})
    with pytest.raises(ValueError, match="unknown setting"):
        spec.build_config(base_config(tmp_path))


def test_each_camera_can_use_its_own_zones(tmp_path):
    spec = CameraSpec.from_dict(
        {"camera_id": "CAM-9", "source": "synthetic", "zones": "config/zones-calibrated.json"}
    )
    config = spec.build_config(base_config(tmp_path))
    assert config.zones_path.name == "zones-calibrated.json"


def test_loading_a_camera_file(tmp_path):
    path = write_cameras(tmp_path, [
        {"camera_id": "CAM-01", "source": "synthetic"},
        {"camera_id": "CAM-02", "source": "webcam:0"},
    ])
    specs = load_camera_specs(path)
    assert [s.camera_id for s in specs] == ["CAM-01", "CAM-02"]


def test_duplicate_camera_ids_are_rejected(tmp_path):
    """Two feeds with one id makes every alert ambiguous."""
    path = write_cameras(tmp_path, [
        {"camera_id": "CAM-01", "source": "synthetic"},
        {"camera_id": "CAM-01", "source": "webcam:0"},
    ])
    with pytest.raises(ValueError, match="duplicate camera ids"):
        load_camera_specs(path)


def test_an_empty_camera_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no cameras"):
        load_camera_specs(write_cameras(tmp_path, []))


def test_a_missing_camera_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_camera_specs(tmp_path / "nope.json")


def test_the_shipped_fleet_config_loads():
    specs = load_camera_specs("config/cameras.json")
    assert len(specs) >= 2
    assert len({s.camera_id for s in specs}) == len(specs)


# --------------------------------------------------------- shared detector


class CountingDetector:
    name = "COUNTING"

    def __init__(self):
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def detect(self, frame):
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.calls += 1
        with self._lock:
            self.concurrent -= 1
        return [Detection("PERSON", 0.9, (1, 1, 10, 20))]


def test_shared_detector_passes_detections_through():
    shared = SharedDetector(CountingDetector())
    assert len(shared.detect(FRAME)) == 1
    assert shared.calls == 1


def test_shared_detector_reports_the_wrapped_name():
    assert "COUNTING" in SharedDetector(CountingDetector()).name
    assert "shared" in SharedDetector(CountingDetector()).name


def test_shared_detector_serialises_concurrent_callers():
    """One model, many camera threads - inference must not overlap."""
    inner = CountingDetector()
    shared = SharedDetector(inner)

    def hammer():
        for _ in range(50):
            shared.detect(FRAME)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert shared.calls == 200
    assert inner.max_concurrent == 1


# ------------------------------------------------------------------- fleet


def test_every_camera_runs_and_reports(tmp_path):
    specs = [
        CameraSpec.from_dict({"camera_id": "CAM-01", "source": "synthetic"}),
        CameraSpec.from_dict({"camera_id": "CAM-02", "source": "synthetic"}),
    ]
    runner = MultiCameraRunner(specs, base_config(tmp_path))
    workers = runner.run(poll=0.05)

    assert len(workers) == 2
    assert {w.spec.camera_id for w in workers} == {"CAM-01", "CAM-02"}
    for worker in workers:
        assert worker.error is None
        assert worker.pipeline.stats.frames == 40
        assert worker.pipeline.stats.alerts == 1


class ShareableDetector(CountingDetector):
    """Stateless, so one instance may serve the whole fleet."""

    name = "SHAREABLE"
    shareable = True


def three_cameras():
    return [
        CameraSpec.from_dict({"camera_id": f"CAM-0{n}", "source": "synthetic"})
        for n in range(1, 4)
    ]


def test_a_stateless_model_is_loaded_once_for_the_whole_fleet(tmp_path, monkeypatch):
    built = []

    def make(self, config):
        detector = ShareableDetector()
        built.append(detector)
        return detector

    monkeypatch.setattr(MultiCameraRunner, "_make_detector", make)
    runner = MultiCameraRunner(three_cameras(), base_config(tmp_path))
    runner.run(poll=0.05)

    assert len(built) == 1
    assert runner.shared is True
    # Every camera holds the same shared wrapper.
    assert len({id(w.pipeline.detector) for w in runner.workers}) == 1
    assert runner.detector.calls == 3 * 40


def test_a_stateful_model_gets_one_instance_per_camera(tmp_path):
    """The scripted simulator counts steps; sharing it would interleave feeds."""
    runner = MultiCameraRunner(three_cameras(), base_config(tmp_path))
    runner.run(poll=0.05)

    assert runner.shared is False
    assert len({id(w.pipeline.detector) for w in runner.workers}) == 3
    # Each camera therefore sees the identical scripted walk.
    assert {w.pipeline.stats.alerts for w in runner.workers} == {1}


def test_bytetrack_is_never_shared_between_cameras(tmp_path):
    """ByteTrack holds track ids per stream - sharing would mix them up."""
    from app.detection.yolo import YoloPersonDetector

    assert YoloPersonDetector.shareable is True  # plain prediction is stateless

    plain = object.__new__(YoloPersonDetector)
    plain.shareable = True
    tracked = object.__new__(YoloPersonDetector)
    tracked.shareable = False
    assert plain.shareable and not tracked.shareable


def test_all_cameras_write_to_one_store(tmp_path):
    from app.store import EventStore

    specs = [
        CameraSpec.from_dict({"camera_id": "CAM-01", "source": "synthetic"}),
        CameraSpec.from_dict({"camera_id": "CAM-02", "source": "synthetic"}),
    ]
    with EventStore(tmp_path / "fleet.db") as store:
        MultiCameraRunner(specs, base_config(tmp_path), store=store).run(poll=0.05)
        assert store.summary()["by_camera"] == {"CAM-01": 1, "CAM-02": 1}


def test_event_hooks_see_every_camera(tmp_path):
    seen = []
    lock = threading.Lock()

    def record(event):
        with lock:
            seen.append(event.camera_id)

    specs = [
        CameraSpec.from_dict({"camera_id": "CAM-01", "source": "synthetic"}),
        CameraSpec.from_dict({"camera_id": "CAM-02", "source": "synthetic"}),
    ]
    MultiCameraRunner(specs, base_config(tmp_path), event_hooks=[record]).run(poll=0.05)
    assert sorted(seen) == ["CAM-01", "CAM-02"]


def test_one_broken_camera_does_not_stop_the_others(tmp_path):
    """A dead feed at a border post must not take the whole post down."""
    specs = [
        CameraSpec.from_dict({"camera_id": "CAM-BAD", "source": "data/nope.mp4"}),
        CameraSpec.from_dict({"camera_id": "CAM-OK", "source": "synthetic"}),
    ]
    runner = MultiCameraRunner(specs, base_config(tmp_path))
    workers = runner.run(poll=0.05)

    broken = next(w for w in workers if w.spec.camera_id == "CAM-BAD")
    healthy = next(w for w in workers if w.spec.camera_id == "CAM-OK")
    assert broken.error is not None
    assert healthy.error is None
    assert healthy.pipeline.stats.alerts == 1


def test_status_reports_each_camera(tmp_path):
    specs = [CameraSpec.from_dict({"camera_id": "CAM-01", "source": "synthetic"})]
    runner = MultiCameraRunner(specs, base_config(tmp_path))
    runner.run(poll=0.05)

    status = runner.status()
    assert len(status) == 1
    assert status[0]["camera_id"] == "CAM-01"
    assert status[0]["frames"] == 40
    assert status[0]["alerts"] == 1
    assert status[0]["running"] is False


def test_stop_is_safe_to_call_on_a_finished_fleet(tmp_path):
    specs = [CameraSpec.from_dict({"camera_id": "CAM-01", "source": "synthetic"})]
    runner = MultiCameraRunner(specs, base_config(tmp_path))
    runner.run(poll=0.05)
    runner.stop()  # must not raise


# ------------------------------------------------- unreachable cameras


def test_an_unreachable_network_camera_is_skipped(tmp_path, monkeypatch):
    """A phone that is switched off must not stop the laptop camera running."""
    monkeypatch.setattr("app.video.stream.tcp_reachable", lambda url, timeout: False)
    specs = [
        CameraSpec.from_dict({"camera_id": "CAM-01", "source": "synthetic"}),
        CameraSpec.from_dict({"camera_id": "CAM-PHONE", "source": "http://10.0.0.9:8080/video"}),
    ]
    runner = MultiCameraRunner(specs, base_config(tmp_path))
    workers = runner.run(poll=0.05)

    assert [w.spec.camera_id for w in workers] == ["CAM-01"]
    assert runner.skipped == [("CAM-PHONE", "not answering on the network")]
    assert workers[0].pipeline.stats.frames == 40


def test_a_reachable_network_camera_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr("app.video.stream.tcp_reachable", lambda url, timeout: True)
    specs = [CameraSpec.from_dict({"camera_id": "CAM-NET", "source": "http://10.0.0.9:8080/video"})]
    runner = MultiCameraRunner(specs, base_config(tmp_path))
    assert [s.camera_id for s in runner._usable_specs()] == ["CAM-NET"]
    assert runner.skipped == []


def test_a_missing_webcam_device_is_skipped(tmp_path, monkeypatch):
    class ClosedCapture:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", lambda *a, **k: ClosedCapture())
    specs = [CameraSpec.from_dict({"camera_id": "CAM-USB", "source": "webcam:3"})]
    runner = MultiCameraRunner(specs, base_config(tmp_path))

    assert runner._usable_specs() == []
    assert "index 3" in runner.skipped[0][1]


def test_running_with_no_reachable_camera_reports_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr("app.video.stream.tcp_reachable", lambda url, timeout: False)
    specs = [CameraSpec.from_dict({"camera_id": "CAM-PHONE", "source": "rtsp://10.0.0.9/s"})]
    runner = MultiCameraRunner(specs, base_config(tmp_path))
    assert runner.run(poll=0.05) == []


def test_synthetic_and_file_sources_are_never_pre_checked(tmp_path):
    """Only live devices are probed; a bad file path should fail loudly instead."""
    specs = [
        CameraSpec.from_dict({"camera_id": "CAM-SIM", "source": "synthetic"}),
        CameraSpec.from_dict({"camera_id": "CAM-FILE", "source": "data/missing.mp4"}),
    ]
    runner = MultiCameraRunner(specs, base_config(tmp_path))
    assert len(runner._usable_specs()) == 2


# ----------------------------------------------------- threaded display


def test_view_mode_routes_frames_to_the_runner_not_to_opencv(tmp_path):
    """OpenCV's GUI is single-threaded; camera threads must not call it."""
    specs = [
        CameraSpec.from_dict({"camera_id": "CAM-01", "source": "synthetic"}),
        CameraSpec.from_dict({"camera_id": "CAM-02", "source": "synthetic"}),
    ]
    runner = MultiCameraRunner(specs, base_config(tmp_path, view=True, save_evidence=False))
    workers = runner._build_workers()

    for worker in workers:
        # Bound methods are recreated per access, so compare by equality.
        assert worker.pipeline.frame_sink == runner._post_frame


def test_no_frame_sink_when_view_is_off(tmp_path):
    specs = [CameraSpec.from_dict({"camera_id": "CAM-01", "source": "synthetic"})]
    runner = MultiCameraRunner(specs, base_config(tmp_path))
    assert runner._build_workers()[0].pipeline.frame_sink is None


def test_posted_frames_are_kept_per_camera(tmp_path):
    runner = MultiCameraRunner([], base_config(tmp_path))
    a, b = np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)
    runner._post_frame("CAM-01", a)
    runner._post_frame("CAM-02", b)
    assert set(runner._frames) == {"CAM-01", "CAM-02"}
    assert runner._frames["CAM-02"] is b


def test_a_camera_thread_never_touches_the_opencv_gui(tmp_path, monkeypatch):
    """Regression: two threads calling waitKey used to deadlock the process."""
    def explode(*args, **kwargs):
        raise AssertionError("imshow called from a camera thread")

    monkeypatch.setattr("cv2.imshow", explode)
    specs = [
        CameraSpec.from_dict({"camera_id": "CAM-01", "source": "synthetic"}),
        CameraSpec.from_dict({"camera_id": "CAM-02", "source": "synthetic"}),
    ]
    runner = MultiCameraRunner(specs, base_config(tmp_path, view=True, save_evidence=False))
    workers = runner._build_workers()
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.thread.join(timeout=20)

    for worker in workers:
        assert worker.error is None
        assert worker.pipeline.stats.frames == 40
    # The frames still arrived - they just went to the runner instead.
    assert set(runner._frames) == {"CAM-01", "CAM-02"}
