import inspect
import pytest
from game_agent.controller import BrowserController
from game_agent.vlm import VLMClient
from game_agent.observation import Observation
from game_agent.schema import ClickAction

def test_controller_methods_are_async_and_raise():
    c = BrowserController()
    assert inspect.iscoroutinefunction(c.open)
    assert inspect.iscoroutinefunction(c.execute)
    assert inspect.iscoroutinefunction(c.release_all)
    assert inspect.iscoroutinefunction(c.screenshot)

async def test_controller_execute_not_implemented():
    with pytest.raises(NotImplementedError):
        await BrowserController().execute(ClickAction(type="click", x=1, y=1))

async def test_vlm_decide_not_implemented():
    obs = Observation(observation_id="o", screenshot_path=None, viewport_width=1,
                      viewport_height=1, device_scale_factor=1.0, timestamp=0.0)
    with pytest.raises(NotImplementedError):
        await VLMClient().decide(obs, goal="win", remaining_actions=10)

def test_release_all_documented_contract():
    assert "release" in (BrowserController.release_all.__doc__ or "").lower()
