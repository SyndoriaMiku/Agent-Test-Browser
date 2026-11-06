import pytest
from game_agent.keys import to_playwright_key

@pytest.mark.parametrize("agent,expected", [
    ("up", "ArrowUp"), ("down", "ArrowDown"), ("left", "ArrowLeft"),
    ("right", "ArrowRight"), ("space", "Space"), ("enter", "Enter"),
    ("escape", "Escape"), ("shift", "Shift"), ("tab", "Tab"),
])
def test_maps_agent_names(agent, expected):
    assert to_playwright_key(agent) == expected

def test_single_char_passthrough():
    assert to_playwright_key("a") == "a"
    assert to_playwright_key("W") == "W"

def test_unknown_name_passthrough():
    # Unknown names are returned unchanged so Playwright reports the error.
    assert to_playwright_key("f13") == "f13"
