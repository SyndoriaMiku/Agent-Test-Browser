import math
import pytest
from pydantic import ValidationError
from game_agent.schema import Decision, ClickAction, HoldAction, UncertainAction

def _decision(**over):
    base = {"state": "menu", "reason": "start visible", "confidence": 0.9,
            "action": {"type": "click", "x": 640, "y": 620}}
    base.update(over)
    return base

def test_valid_click_decision():
    d = Decision.model_validate(_decision())
    assert isinstance(d.action, ClickAction)
    assert d.action.button.value == "left"
    assert d.action.click_count == 1

def test_discriminator_picks_hold():
    d = Decision.model_validate(_decision(action={"type": "hold", "key": "space",
                                                  "duration_ms": 500}))
    assert isinstance(d.action, HoldAction)

def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "click", "x": 1, "y": 2,
                                                  "evil": True}))

def test_unknown_type_rejected():
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "exec", "code": "rm -rf /"}))

def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(confidence=1.5))
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(confidence=math.nan))

def test_click_count_bounds():
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "click", "x": 1, "y": 2,
                                                  "click_count": 4}))

def test_decision_is_frozen():
    d = Decision.model_validate(_decision())
    with pytest.raises(ValidationError):
        d.confidence = 0.1  # type: ignore[misc]
