from game_agent.config import Settings, load_settings

def test_defaults():
    s = load_settings(env={})
    assert s.viewport_width == 1280
    assert s.viewport_height == 720
    assert s.confidence_threshold == 0.55
    assert "left" in s.allowed_buttons

def test_env_override():
    s = load_settings(env={"GAME_AGENT_VIEWPORT_WIDTH": "1920",
                           "GAME_AGENT_CONFIDENCE_THRESHOLD": "0.8"})
    assert s.viewport_width == 1920
    assert s.confidence_threshold == 0.8

def test_settings_is_frozen():
    s = Settings()
    try:
        s.viewport_width = 100  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Settings must be immutable")
