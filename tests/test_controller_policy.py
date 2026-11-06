import dataclasses

import pytest
from game_agent.config import Settings
from game_agent.controller import (ActionExecutionError, ActionNotAllowed,
                                   BrowserOptions, ControllerError, assert_executable)
from game_agent.schema import (ClickAction, DoneAction, HoldAction, MoveAction,
                               PressAction, UncertainAction, WaitAction)

S = Settings(viewport_width=1280, viewport_height=720, max_wait_ms=5000,
             max_hold_ms=2000, allowed_buttons=frozenset({"left"}),
             allowed_keys=frozenset({"w", "up"}))

def test_options_defaults_and_frozen():
    o = BrowserOptions()
    assert o.headless is False
    assert o.user_data_dir == ".browser-profile"
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.headless = True  # type: ignore[misc]

def test_exception_hierarchy():
    assert issubclass(ActionNotAllowed, ControllerError)
    assert issubclass(ActionExecutionError, ControllerError)

def test_accepts_valid_click():
    assert_executable(ClickAction(type="click", x=10, y=10), S)

def test_rejects_done_and_uncertain():
    with pytest.raises(ActionNotAllowed):
        assert_executable(DoneAction(type="done", summary="won"), S)
    with pytest.raises(ActionNotAllowed):
        assert_executable(UncertainAction(type="uncertain", reason="hmm"), S)

def test_rejects_key_not_allowed():
    with pytest.raises(ActionNotAllowed):
        assert_executable(PressAction(type="press", key="q"), S)

def test_rejects_button_not_allowed():
    with pytest.raises(ActionNotAllowed):
        assert_executable(ClickAction(type="click", x=1, y=1, button="right"), S)

def test_rejects_out_of_bounds():
    with pytest.raises(ActionNotAllowed):
        assert_executable(MoveAction(type="move", x=1281, y=1), S)

def test_rejects_over_duration_caps():
    with pytest.raises(ActionNotAllowed):
        assert_executable(WaitAction(type="wait", duration_ms=5001), S)
    with pytest.raises(ActionNotAllowed):
        assert_executable(HoldAction(type="hold", key="w", duration_ms=2001), S)
