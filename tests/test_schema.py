import math
import pytest
from pydantic import ValidationError
from game_agent.schema import (
    Decision,
    ClickAction,
    DragAction,
    HoldAction,
    UncertainAction,
    WheelAction,
)

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

def test_wheel_delta_fields_parse():
    d = Decision.model_validate(_decision(action={"type": "wheel", "delta_y": 100}))
    assert isinstance(d.action, WheelAction)
    assert d.action.delta_x == 0
    assert d.action.delta_y == 100

def test_uncertain_uses_reason():
    d = Decision.model_validate(_decision(action={"type": "uncertain", "reason": "unknown"}))
    assert isinstance(d.action, UncertainAction)
    assert d.action.reason == "unknown"

def test_drag_shape_and_duration():
    d = Decision.model_validate(_decision(action={"type": "drag", "start_x": 1, "start_y": 2,
                                                  "end_x": 3, "end_y": 4, "duration_ms": 250}))
    assert isinstance(d.action, DragAction)
    assert d.action.duration_ms == 250

def test_wrong_numeric_types_rejected():
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "click", "x": "640", "y": 620}))
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "click", "x": 1, "y": 2,
                                                  "click_count": True}))
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(confidence="0.5"))
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(confidence=True))

def test_hold_release_rejects_both_key_and_button():
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "hold", "key": "space",
                                                  "button": "left", "duration_ms": 500}))
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "release", "key": "space",
                                                  "button": "left"}))

def test_hold_release_rejects_neither_key_nor_button():
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "hold", "duration_ms": 500}))
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "release"}))

def test_empty_key_rejected():
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "hold", "key": "",
                                                  "duration_ms": 500}))

def test_zero_duration_rejected():
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "hold", "key": "space",
                                                  "duration_ms": 0}))
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "wait", "duration_ms": 0}))
