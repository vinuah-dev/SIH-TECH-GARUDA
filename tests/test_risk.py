from datetime import datetime

import pytest

from app.detection.base import Detection
from app.context.scene import SceneContext
from app.risk.engine import RiskEngine
from app.zones.manager import Zone

NOON = datetime(2026, 8, 25, 12, 0)
MIDNIGHT = datetime(2026, 8, 25, 23, 30)

RESTRICTED = Zone("RESTRICTED", "RESTRICTED", [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0)], base_risk=60)
WATCH = Zone("WATCH", "WATCH", [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0)], base_risk=30)


def person(confidence=0.94):
    return Detection("PERSON", confidence, (10, 10, 60, 200), track_id=1)


def context(when, force_night=None):
    return SceneContext.build("CAM-01", when, force_night=force_night)


def test_night_window_wraps_past_midnight():
    assert context(datetime(2026, 8, 25, 23, 0)).is_night
    assert context(datetime(2026, 8, 25, 3, 0)).is_night
    assert not context(NOON).is_night


def test_restricted_night_intrusion_scores_high():
    """Zone (60) + confidence (10) + night presence (6), with no behaviours yet."""
    assessment = RiskEngine().assess(person(), RESTRICTED, context(MIDNIGHT))
    assert assessment.score == 76
    assert assessment.severity == "HIGH"


def test_behaviours_add_their_points_to_the_score():
    from app.behaviour.engine import Behaviour

    engine = RiskEngine()
    plain = engine.assess(person(), RESTRICTED, context(MIDNIGHT))
    with_behaviour = engine.assess(
        person(), RESTRICTED, context(MIDNIGHT), None,
        [Behaviour("LOITERING", 12, "12s inside RESTRICTED")],
    )
    assert with_behaviour.score == plain.score + 12
    assert any(f.label == "LOITERING" for f in with_behaviour.factors)


def test_behaviour_points_remain_attributable():
    from app.behaviour.engine import Behaviour

    assessment = RiskEngine().assess(
        person(), RESTRICTED, context(MIDNIGHT), None,
        [Behaviour("LOITERING", 12, "x"), Behaviour("NIGHT MOVEMENT", 4, "y")],
    )
    assert sum(f.points for f in assessment.factors) == assessment.score


def test_daytime_scores_lower_than_night():
    engine = RiskEngine()
    day = engine.assess(person(), RESTRICTED, context(NOON))
    night = engine.assess(person(), RESTRICTED, context(MIDNIGHT))
    assert night.score > day.score


def test_watch_zone_scores_below_restricted():
    engine = RiskEngine()
    assert engine.assess(person(), WATCH, context(MIDNIGHT)).score < \
        engine.assess(person(), RESTRICTED, context(MIDNIGHT)).score


def test_low_confidence_reduces_score():
    engine = RiskEngine()
    assert engine.assess(person(0.41), RESTRICTED, context(NOON)).score < \
        engine.assess(person(0.94), RESTRICTED, context(NOON)).score


def test_score_is_clamped_to_0_100():
    engine = RiskEngine()
    huge = Zone("X", "RESTRICTED", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], base_risk=200)
    assert engine.assess(person(), huge, context(MIDNIGHT)).score == 100


def test_every_point_is_attributable_to_a_factor():
    """Explainability guarantee: the score is exactly the sum of its factors."""
    assessment = RiskEngine().assess(person(), RESTRICTED, context(MIDNIGHT))
    assert sum(f.points for f in assessment.factors) == assessment.score
    assert len(assessment.factors) == 3


@pytest.mark.parametrize(
    "score,severity",
    [(95, "CRITICAL"), (90, "CRITICAL"), (89, "HIGH"), (70, "HIGH"), (45, "MEDIUM"), (10, "LOW")],
)
def test_severity_thresholds(score, severity):
    assert RiskEngine().severity(score) == severity


def test_reason_mentions_the_zone():
    assessment = RiskEngine().assess(person(), RESTRICTED, context(MIDNIGHT))
    assert "Restricted-zone entry" in assessment.reason
    assert "night-time" in assessment.reason
