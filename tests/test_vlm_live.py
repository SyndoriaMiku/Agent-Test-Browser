"""Live end-to-end test: real browser screenshot through the real VLM API.

Opt-in twice over — the ``live`` and ``browser`` markers keep it out of the
default (offline) suite.  Run it with::

    .venv\\Scripts\\python -m pytest -m "live and browser" tests/test_vlm_live.py -v

It skips unless both ``NVIDIA_API_KEY`` and ``NVIDIA_MODEL`` resolve from the
process environment or ``.env`` — the guard reads them through
``load_vlm_config``, exactly as the client does (see the Vision model section
in the README).  A missing Chromium binary skips too, so a fresh checkout is
never red.
"""

import pathlib

import pytest
from playwright.async_api import Error as PlaywrightError

from game_agent.config import Settings
from game_agent.controller import BrowserController, BrowserOptions
from game_agent.schema import Decision
from game_agent.vlm import VLMClient, load_vlm_config
from tests.test_browser_controller import _close_quietly, _is_missing_executable

pytestmark = [pytest.mark.live, pytest.mark.browser]
DEMO = pathlib.Path(__file__).resolve().parents[1] / "examples" / "demo.html"


async def test_live_decide_on_demo_screenshot(tmp_path):
    config = load_vlm_config()
    if not (config.api_key and config.model):
        pytest.skip(
            "set NVIDIA_API_KEY and NVIDIA_MODEL in the environment or .env "
            "to run the live test"
        )
    controller = BrowserController(Settings(), options=BrowserOptions(
        headless=True, user_data_dir=str(tmp_path / "p"), screenshot_dir=str(tmp_path / "s")))
    try:
        await controller.open()
    except PlaywrightError as exc:
        await _close_quietly(controller)
        if _is_missing_executable(exc):
            reason = str(exc).splitlines()[0]
            pytest.skip(
                f"Chromium is not installed for Playwright: {reason}. "
                "Run `.venv\\Scripts\\python -m playwright install chromium`."
            )
        raise
    try:
        await controller.goto(DEMO.as_uri())
        obs = await controller.screenshot()
        decision = await VLMClient(config).decide(
            obs, goal="Find and click the Start button.", remaining_actions=3)
    finally:
        await controller.close()
    assert isinstance(decision, Decision)
    assert decision.observation_id == obs.observation_id
