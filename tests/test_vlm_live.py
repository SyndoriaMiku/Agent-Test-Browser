"""Live end-to-end test: real browser screenshot through the real VLM API.

Opt-in twice over — the ``live`` and ``browser`` markers keep it out of the
default (offline) suite.  Run it with::

    .venv\\Scripts\\python -m pytest -m "live and browser" tests/test_vlm_live.py -v

It skips unless both ``NVIDIA_API_KEY`` and ``NVIDIA_MODEL`` are set (see the
Vision model section in the README), so a fresh checkout is never red.
"""

import os
import pathlib

import pytest

from game_agent.config import Settings
from game_agent.controller import BrowserController, BrowserOptions
from game_agent.schema import Decision
from game_agent.vlm import VLMClient

pytestmark = [pytest.mark.live, pytest.mark.browser]
DEMO = pathlib.Path(__file__).resolve().parents[1] / "examples" / "demo.html"


async def test_live_decide_on_demo_screenshot(tmp_path):
    if not (os.environ.get("NVIDIA_API_KEY") and os.environ.get("NVIDIA_MODEL")):
        pytest.skip("set NVIDIA_API_KEY and NVIDIA_MODEL to run the live test")
    controller = BrowserController(Settings(), options=BrowserOptions(
        headless=True, user_data_dir=str(tmp_path / "p"), screenshot_dir=str(tmp_path / "s")))
    await controller.open()
    try:
        await controller.goto(DEMO.as_uri())
        obs = await controller.screenshot()
        decision = await VLMClient().decide(obs, goal="Find and click the Start button.", remaining_actions=3)
    finally:
        await controller.close()
    assert isinstance(decision, Decision)
    assert decision.observation_id == obs.observation_id
