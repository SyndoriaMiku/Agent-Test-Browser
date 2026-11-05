import pytest
from game_agent.config import Settings
from game_agent.schema import Decision
from game_agent.session_state import StateSnapshot
from game_agent.validation import RejectedReason, validate

SETTINGS = Settings(viewport_width=1280, viewport_height=720,
                    min_action_interval_ms=200, max_actions_per_session=2,
                    max_hold_ms=1000, max_wheel_delta=100, confidence_threshold=0.5)

def _snap(count=0, hold=0, fails=0, last=None):
    return StateSnapshot(action_count=count, accumulated_hold_ms=hold,
                         consecutive_failures=fails, last_action_at=last)

def _dec(action, conf=0.9):
    return Decision.model_validate({"state":"s","reason":"r","confidence":conf,"action":action})

def test_accepts_valid_click():
    assert validate(_dec({"type":"click","x":10,"y":10}), _snap(), now=10.0, settings=SETTINGS).accepted

def test_accepts_coordinates_on_the_closed_edge():
    r = validate(_dec({"type":"click","x":1280,"y":720}), _snap(), now=10.0, settings=SETTINGS)
    assert r.accepted

def test_rejects_out_of_bounds():
    assert validate(_dec({"type":"click","x":-1,"y":10}), _snap(), now=10.0, settings=SETTINGS).reason == RejectedReason.OUT_OF_BOUNDS
    assert validate(_dec({"type":"click","x":10,"y":721}), _snap(), now=10.0, settings=SETTINGS).reason == RejectedReason.OUT_OF_BOUNDS

def test_rejects_key_not_in_allowlist():
    assert validate(_dec({"type":"press","key":"f13"}), _snap(), now=10.0, settings=SETTINGS).reason == RejectedReason.NOT_ALLOWED

def test_rate_limit_boundary():
    # last action at t=10.0, min interval 200ms. Use binary-exact deltas (0.125s, 0.25s)
    # so the boundary assertion is not corrupted by float rounding.
    assert validate(_dec({"type":"move","x":1,"y":1}), _snap(last=10.0), now=10.125, settings=SETTINGS).reason == RejectedReason.RATE_LIMITED
    assert validate(_dec({"type":"move","x":1,"y":1}), _snap(last=10.0), now=10.25, settings=SETTINGS).accepted

def test_budget_boundary_off_by_one():
    assert validate(_dec({"type":"move","x":1,"y":1}), _snap(count=1), now=10.0, settings=SETTINGS).accepted
    assert validate(_dec({"type":"move","x":1,"y":1}), _snap(count=2), now=10.0, settings=SETTINGS).reason == RejectedReason.BUDGET_EXCEEDED

def test_hold_accumulation_guard():
    assert validate(_dec({"type":"hold","key":"w","duration_ms":500}), _snap(hold=500), now=10.0, settings=SETTINGS).accepted
    assert validate(_dec({"type":"hold","key":"w","duration_ms":600}), _snap(hold=500), now=10.0, settings=SETTINGS).reason == RejectedReason.LIMIT_EXCEEDED

def test_wheel_delta_guard():
    assert validate(_dec({"type":"wheel","delta_y":100}), _snap(), now=10.0, settings=SETTINGS).accepted
    assert validate(_dec({"type":"wheel","delta_y":101}), _snap(), now=10.0, settings=SETTINGS).reason == RejectedReason.LIMIT_EXCEEDED

def test_low_confidence_rejected():
    assert validate(_dec({"type":"click","x":1,"y":1}, conf=0.4), _snap(), now=10.0, settings=SETTINGS).reason == RejectedReason.LOW_CONFIDENCE

def test_done_and_uncertain_bypass_rate_and_budget():
    assert validate(_dec({"type":"done","summary":"won"}), _snap(count=99, last=10.0), now=10.0, settings=SETTINGS).accepted
    assert validate(_dec({"type":"uncertain","reason":"unknown"}), _snap(count=99, last=10.0), now=10.0, settings=SETTINGS).accepted
