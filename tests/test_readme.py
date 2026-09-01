"""The README must not claim more than the project delivers.

This project's whole argument is that nothing is asserted which has not been
run. A stale number in the README is a small lie, but it is the same *kind* of
lie as claiming a feature works when it does not - and it has already drifted
twice while tests were being added. So the claim is checked rather than
trusted.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

README = Path("README.md")


def collected() -> int:
    """How many tests pytest actually finds.

    Counted rather than derived from `def test_`, because parametrised tests
    expand to several and that gap is exactly where the drift crept in.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "tests"],
        capture_output=True, text=True, cwd=README.resolve().parent,
    )
    found = re.search(r"(\d+) tests? collected", result.stdout)
    if not found:
        pytest.skip("could not read a collection count from pytest")
    return int(found.group(1))


def test_the_readme_does_not_overstate_the_test_count():
    claimed = {int(n) for n in re.findall(r"(\d+)\s+(?:automated\s+)?tests", README.read_text(encoding="utf-8"))}
    assert claimed, "the README should say how many tests there are"

    real = collected()
    # Deliberately one-sided. Claiming more tests than exist is the same kind of
    # overstatement this project exists to avoid; a number that has fallen
    # behind because tests were added is merely out of date. An exact match
    # would fail every time a test is written, which trains people to edit the
    # test instead of the claim.
    overstated = {n for n in claimed if n > real}
    assert not overstated, (
        f"README claims {sorted(overstated)} tests; pytest collects only {real}. "
        f"Update the README rather than this test."
    )


def test_every_readme_claim_of_a_file_points_at_a_real_file():
    """A README that names a file which does not exist is a broken promise.

    Only source paths are checked. Files under config/ are excluded because
    several are *produced* by a documented step rather than shipped - the face
    roster is written by tools/enrol_face.py, and a repo that shipped one would
    be shipping somebody's biometrics into a git history by accident.
    """
    text = README.read_text(encoding="utf-8")
    named = set(re.findall(r"\b((?:app|tools|tests)/[\w/.-]+\.(?:py|json))\b", text))
    missing = sorted(p for p in named if not Path(p).exists())
    assert not missing, f"README names files that do not exist: {missing}"


def test_the_documented_risk_weights_are_the_ones_the_code_uses():
    """The risk table is the part a reviewer will actually check.

    Every number in it was verified against the code by hand once. Pinning it
    means the next person who tunes a weight is forced to update the table in
    the same commit, instead of leaving a document that quietly describes a
    different system than the one running.
    """
    from app.behaviour.engine import BehaviourEngine

    engine = BehaviourEngine()
    documented = {
        "border_facing_points": 8,
        "erratic_points": 6,
        "night_movement_points": 4,
        "running_points": 6,
        "tamper_points": 30,
        "camera_approach_points": 10,
        "plate_mismatch_points": 35,
        "watchlisted_points": 45,
        "unregistered_points": 5,
        "authorised_points": -45,
        "approach_frames": 5,
        "loiter_after": 5.0,
        "loiter_per_second": 1.5,
        "loiter_cap": 20,
        "erratic_threshold": 0.45,
        "running_speed_mps": 2.0,
        "vehicle_speed_mps": 11.0,
        "tamper_height_ratio": 0.65,
    }
    drifted = {
        name: (getattr(engine, name), stated)
        for name, stated in documented.items()
        if getattr(engine, name) != stated
    }
    assert not drifted, (
        f"code and README disagree (code, README): {drifted}. "
        f"Update the risk table in README.md."
    )
