import pytest
from game_agent.observation import Observation

def test_observation_fields_and_immutability():
    o = Observation(observation_id="obs-1", screenshot_path="screenshots/1.png",
                    viewport_width=1280, viewport_height=720,
                    device_scale_factor=1.0, timestamp=12.5)
    assert o.observation_id == "obs-1"
    assert o.viewport_width == 1280
    with pytest.raises(Exception):
        o.observation_id = "obs-2"  # type: ignore[misc]

def test_screenshot_path_optional():
    o = Observation(observation_id="obs-2", screenshot_path=None,
                    viewport_width=1280, viewport_height=720,
                    device_scale_factor=1.0, timestamp=0.0)
    assert o.screenshot_path is None
