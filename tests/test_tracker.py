from app.detection.base import Detection, iou
from app.detection.tracker import IoUTracker


def box(x, width=40, height=90, y=100):
    return Detection("PERSON", 0.9, (x, y, x + width, y + height))


def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_identity_survives_small_movement():
    tracker = IoUTracker()
    first = tracker.update([box(100)])
    second = tracker.update([box(108)])
    assert first[0].track_id == second[0].track_id == 1


def test_large_jump_creates_a_new_track():
    tracker = IoUTracker()
    tracker.update([box(100)])
    assert tracker.update([box(600)])[0].track_id == 2


def test_two_people_keep_separate_ids():
    tracker = IoUTracker()
    tracker.update([box(100), box(400)])
    result = tracker.update([box(105), box(405)])
    assert {d.track_id for d in result} == {1, 2}


def test_output_order_matches_input_order():
    tracker = IoUTracker()
    tracker.update([box(100), box(400)])
    result = tracker.update([box(405), box(105)])
    assert [d.bbox[0] for d in result] == [405, 105]


def test_track_is_dropped_after_max_missed_frames():
    tracker = IoUTracker(max_missed=2)
    tracker.update([box(100)])
    for _ in range(4):
        tracker.update([])
    assert tracker.update([box(100)])[0].track_id == 2


def test_track_label_is_zero_padded():
    assert box(0).track_label == "P---"
    assert IoUTracker().update([box(0)])[0].track_label == "P001"
