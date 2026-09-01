"""Every vehicle that passed, once each.

Alerts answer "what went wrong". This answers the other question a post
actually gets asked: *what came through here today?* Most vehicles break no
rule, so an alert log never contains them - and a week later that is exactly
the list somebody wants.

The two things that make such a log usable are in tension, and most of these
tests are about holding both at once: one entry per vehicle rather than one per
frame, but a vehicle that genuinely comes back must still appear twice.
"""

import json
from datetime import datetime

import numpy as np
import pytest

from app.anpr.plate_log import PlateLog
from app.detection.base import Detection

NOON = datetime(2026, 8, 27, 12, 0, 0).timestamp()


def car(track_id=1, label="CAR"):
    return Detection(label, 0.9, (100, 100, 260, 190), track_id=track_id)


def crop(value=180):
    """A crop with real texture, so sharpness can rank one against another."""
    image = np.full((40, 140, 3), 40, np.uint8)
    image[10:30, 10:130] = value
    image[15:25, 20:120] = 255 - value
    return image


@pytest.fixture
def log(tmp_path):
    return PlateLog(directory=tmp_path, window=10.0, linger=2.0)


# --------------------------------------------------------- one entry per pass


def test_a_vehicle_in_view_for_many_frames_is_logged_once(log):
    """A vehicle is in view for hundreds of frames; the log is not."""
    for i in range(300):
        log.observe("CAM-01", car(), i * 0.03, NOON + i * 0.03)

    assert log.written == 0, "nothing is written while the vehicle is still in view"
    rows = log.flush_departed(now=60.0)
    assert len(rows) == 1
    assert rows[0]["frames"] == 300


def test_the_entry_records_how_long_the_vehicle_was_in_view(log):
    log.observe("CAM-01", car(), 10.0, NOON)
    log.observe("CAM-01", car(), 14.5, NOON + 4.5)
    row = log.flush_departed(now=30.0)[0]
    assert row["seconds_in_view"] == 4.5


def test_a_vehicle_still_in_view_is_not_written_yet(log):
    log.observe("CAM-01", car(), 10.0, NOON)
    assert log.flush_departed(now=11.0) == [], "it has not left"
    assert log.flush_departed(now=13.0), "it has now"


def test_a_brief_occlusion_does_not_split_one_pass_in_two(log):
    """Passing behind a bus for a moment is one crossing, not two."""
    log.observe("CAM-01", car(), 10.0, NOON)
    log.flush_departed(now=11.5)          # inside the linger window
    log.observe("CAM-01", car(), 11.8, NOON + 1.8)
    rows = log.flush_departed(now=20.0)

    assert log.written == 1
    assert rows[0]["frames"] == 2


def test_everything_still_in_view_is_written_when_the_run_ends(log):
    log.observe("CAM-01", car(1), 10.0, NOON)
    log.observe("CAM-01", car(2), 10.0, NOON)
    assert len(log.close()) == 2


# ------------------------------------------------------- but a return counts


def test_the_same_plate_seen_again_much_later_is_a_second_crossing(log):
    """A car that passes at 09:00 and again at 09:02 is the pattern worth seeing."""
    class Read:
        text, confidence, display = "MH12AB1234", 0.8, "MH12 AB 1234"

    log.observe("CAM-01", car(1), 10.0, NOON, reading=Read())
    log.flush_departed(now=30.0)
    log.observe("CAM-01", car(2), 200.0, NOON + 190, reading=Read())
    log.flush_departed(now=220.0)

    assert log.written == 2
    assert log.suppressed == 0


def test_the_same_plate_inside_the_window_is_one_crossing(log):
    """Two track ids for one car - a split track - must not become two rows."""
    class Read:
        text, confidence, display = "MH12AB1234", 0.8, "MH12 AB 1234"

    log.observe("CAM-01", car(1), 10.0, NOON, reading=Read())
    log.flush_departed(now=13.0)
    log.observe("CAM-01", car(2), 14.0, NOON + 4, reading=Read())
    log.flush_departed(now=17.0)

    assert log.written == 1
    assert log.suppressed == 1


def test_an_unread_vehicle_falls_back_to_its_track_id(log):
    """Most plates are not read, so the log must still work without one."""
    log.observe("CAM-01", car(1), 10.0, NOON)
    log.observe("CAM-01", car(2), 10.0, NOON)
    rows = log.flush_departed(now=30.0)

    assert len(rows) == 2, "two tracks with no plate are two vehicles"
    assert all(row["plate"] is None for row in rows)
    assert all(row["plate_read"] is False for row in rows)


def test_the_window_is_configurable(tmp_path):
    class Read:
        text, confidence, display = "MH12AB1234", 0.8, "MH12 AB 1234"

    patient = PlateLog(directory=tmp_path, window=600.0, linger=2.0)
    patient.observe("CAM-01", car(1), 10.0, NOON, reading=Read())
    patient.flush_departed(now=30.0)
    patient.observe("CAM-01", car(2), 200.0, NOON + 190, reading=Read())
    patient.flush_departed(now=220.0)

    assert patient.written == 1, "inside a ten-minute window this is still one car"


# --------------------------------------------------------------- what it logs


def test_every_vehicle_is_logged_whatever_zone_it_was_in(log):
    """The whole point: a vehicle that broke no rule still passed the post."""
    log.observe("CAM-01", car(), 10.0, NOON)
    row = log.flush_departed(now=30.0)[0]

    assert row["vehicle"] == "CAR"
    assert row["camera_id"] == "CAM-01"
    # Nothing about zones, alerts or risk - this log is not about those.
    assert "zone" not in row and "risk" not in row


def test_people_are_not_vehicles(log):
    log.observe("CAM-01", Detection("PERSON", 0.9, (0, 0, 40, 120), track_id=1),
                10.0, NOON)
    assert log.close() == []


def test_an_untracked_vehicle_is_ignored(log):
    """Without a track there is no way to tell one pass from the next frame."""
    log.observe("CAM-01", Detection("CAR", 0.9, (0, 0, 90, 60)), 10.0, NOON)
    assert log.close() == []


def test_a_read_plate_is_recorded_with_its_confidence(log):
    class Read:
        text, confidence, display = "MH12AB1234", 0.83, "MH12 AB 1234"

    log.observe("CAM-01", car(), 10.0, NOON, reading=Read())
    row = log.flush_departed(now=30.0)[0]

    assert row["plate"] == "MH12AB1234"
    assert row["display"] == "MH12 AB 1234"
    assert row["confidence"] == 0.83
    assert row["plate_read"] is True


def test_the_sharpest_crop_of_the_pass_is_the_one_kept(log):
    """A plate is sharpest somewhere in the middle, which is why writing waits."""
    saved = []

    class Store:
        def save(self, image, event_id, when, camera_id):
            saved.append(image)
            return f"{event_id}.jpg"

    log.evidence = Store()
    blurred = np.full((40, 140, 3), 90, np.uint8)
    log.observe("CAM-01", car(), 10.0, NOON, crop=blurred)
    log.observe("CAM-01", car(), 11.0, NOON + 1, crop=crop())
    log.observe("CAM-01", car(), 12.0, NOON + 2, crop=blurred)
    log.flush_departed(now=30.0)

    assert len(saved) == 1
    assert saved[0].std() > blurred.std(), "the flat crop was kept over the sharp one"


# ------------------------------------------------------------------ the file


def test_the_log_is_written_as_one_json_object_per_line(log, tmp_path):
    log.observe("CAM-01", car(1), 10.0, NOON)
    log.observe("CAM-01", car(2), 10.0, NOON)
    log.flush_departed(now=30.0)

    written = list(tmp_path.glob("passes-*.jsonl"))
    assert len(written) == 1, "one file per day"
    rows = [json.loads(line) for line in written[0].read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["at"].startswith("2026-08-27T12:00")


def test_the_log_can_be_switched_off(tmp_path):
    off = PlateLog(directory=tmp_path, enabled=False)
    off.observe("CAM-01", car(), 10.0, NOON)
    assert off.close() == []
    assert off.flush_departed(now=99.0) == []
    assert not list(tmp_path.glob("passes-*.jsonl"))


def test_suppression_keys_do_not_grow_without_bound(log):
    """A camera runs for weeks; nothing in it may accumulate for weeks."""
    for i in range(400):
        log.observe("CAM-01", car(i), float(i) * 30, NOON + i * 30)
        log.flush_departed(now=float(i) * 30 + 60)

    assert log.written == 400
    assert len(log._recent) < 50, f"suppression table grew to {len(log._recent)}"


# ------------------------------------------------------------- in the pipeline


def test_the_pipeline_logs_passes_without_needing_an_alert(tmp_path):
    from app.config import SurveillanceConfig
    from app.pipeline import SurveillancePipeline

    config = SurveillanceConfig(
        source="synthetic", detector="sim", synthetic_frames=10,
        zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, color=False, quiet=True, print_summary=False,
    )
    pipeline = SurveillancePipeline(config)
    assert pipeline.passes.enabled
    assert pipeline.passes.window == config.plate_log_window


def test_turning_anpr_off_turns_the_pass_log_off_too(tmp_path):
    """There is nothing to log a plate from once plate reading is off."""
    from app.config import SurveillanceConfig
    from app.pipeline import SurveillancePipeline

    config = SurveillanceConfig(
        source="synthetic", detector="sim", synthetic_frames=5, anpr=False,
        zones_path="config/zones.json",
        events_dir=tmp_path / "e", evidence_dir=tmp_path / "v",
        save_clips=False, color=False, quiet=True, print_summary=False,
    )
    assert not SurveillancePipeline(config).passes.enabled
