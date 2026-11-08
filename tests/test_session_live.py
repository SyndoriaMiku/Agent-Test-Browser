"""Live end-to-end test: the real closed-loop session against the real VLM API.

Opt-in twice over -- the ``live`` and ``browser`` markers keep it out of the
default (offline) suite.  Run it with::

    .venv\\Scripts\\python -m pytest -m "live and browser" tests/test_session_live.py -v

It skips unless both ``NVIDIA_API_KEY`` and ``NVIDIA_MODEL`` resolve from the
process environment or ``.env`` -- the guard reads them through
``load_vlm_config``, exactly as the client does (see the Vision model section
in the README).  A missing Chromium binary skips too, so a fresh checkout is
never red.

The session itself is the real one: a real headless Chromium drives
``examples/demo.html``, the real model decides, and the real validator,
budget and failure cap guard every action.  The assertion pins what M3
actually guarantees -- the loop ran to a terminal state: ``DONE`` when the
model reports the goal met, ``NEEDS_USER`` when it cannot continue safely,
``STOPPED`` when it hits a safety limit such as the cycle limit.  It does
not assert task success: the demo page keeps its ``Start`` button on screen
after a click, so a model that keeps re-clicking it burns the cycle limit
instead of finishing the goal.
"""

import pathlib

import pytest
from playwright.async_api import Error as PlaywrightError

from game_agent.config import Settings
from game_agent.controller import BrowserController, BrowserOptions
from game_agent.session import Session
from game_agent.state import State
from game_agent.vlm import VLMClient, load_vlm_config
from tests.test_browser_controller import _close_quietly, _is_missing_executable

pytestmark = [pytest.mark.live, pytest.mark.browser]
DEMO = pathlib.Path(__file__).resolve().parents[1] / "examples" / "demo.html"


async def test_live_session_clicks_demo(tmp_path):
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
        session = Session(controller, VLMClient(config), Settings(), grid_step=100)
        result = await session.run(
            "Click the Start button, then press the up arrow.", max_cycles=6
        )
    finally:
        await controller.close()
    # M3 guarantees safe termination, not task success: the run must reach a
    # terminal state within the cycle budget.  Detecting goal progress (telling
    # "still working" apart from "made no progress") is deferred to M4.
    assert result.state in (State.DONE, State.NEEDS_USER, State.STOPPED)
    assert 1 <= result.cycles <= 6
