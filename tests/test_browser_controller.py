"""Browser tests for the real Playwright-backed controller.

These are opt-in: the ``browser`` marker keeps them out of the default
(offline) suite.  Run them with::

    .venv\\Scripts\\python -m pytest -m browser tests/test_browser_controller.py -v

If the Chromium executable is not installed yet, the tests skip with a clear
message instead of failing, so a fresh checkout is never red.
"""

import pathlib
import struct

import pytest
from playwright.async_api import Error as PlaywrightError

from game_agent.config import Settings
from game_agent.controller import BrowserController, BrowserOptions
from game_agent.observation import Observation

pytestmark = pytest.mark.browser
DEMO = pathlib.Path(__file__).resolve().parents[1] / "examples" / "demo.html"

_MISSING_BROWSER_HINTS = ("executable doesn't exist", "playwright install")


def _is_missing_executable(exc: Exception) -> bool:
    """True when *exc* says the browser binary itself is not installed."""
    message = str(exc).lower()
    return any(hint in message for hint in _MISSING_BROWSER_HINTS)


async def _close_quietly(controller: BrowserController) -> None:
    """Best-effort teardown for a controller that never finished opening."""
    try:
        await controller.close()
    except Exception:
        pass


def _png_size(path: str) -> tuple[int, int]:
    """Read (width, height) in pixels from a PNG's IHDR chunk."""
    data = pathlib.Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


@pytest.fixture
async def controller(tmp_path):
    opts = BrowserOptions(headless=True, user_data_dir=str(tmp_path / "profile"),
                          screenshot_dir=str(tmp_path / "shots"))
    c = BrowserController(Settings(), options=opts)
    try:
        await c.open()
    except PlaywrightError as exc:
        await _close_quietly(c)
        if _is_missing_executable(exc):
            reason = str(exc).splitlines()[0]
            pytest.skip(
                f"Chromium is not installed for Playwright: {reason}. "
                "Run `.venv\\Scripts\\python -m playwright install chromium`."
            )
        raise
    try:
        yield c
    finally:
        await c.close()


async def test_screenshot_dimensions(controller):
    obs = await controller.screenshot()
    assert isinstance(obs, Observation)
    assert obs.viewport_width == 1280 and obs.viewport_height == 720
    assert obs.screenshot_path and pathlib.Path(obs.screenshot_path).exists()


async def test_goto_demo_and_read_state(controller):
    await controller.goto(DEMO.as_uri())
    state = await controller.page.evaluate("window.__demoState")
    assert state == {"clicks": 0, "lastKey": None}


async def test_screenshot_png_matches_viewport(controller):
    obs = await controller.screenshot()
    assert obs.screenshot_path
    assert _png_size(obs.screenshot_path) == (1280, 720)


async def test_screenshot_retries_transient_capture_failure(controller, monkeypatch):
    original = controller.page.screenshot
    calls: list[int] = []

    async def flaky_screenshot(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise PlaywrightError(
                "Page.screenshot: Protocol error (Page.captureScreenshot): "
                "Unable to capture screenshot"
            )
        return await original(*args, **kwargs)

    monkeypatch.setattr(controller.page, "screenshot", flaky_screenshot)
    obs = await controller.screenshot()
    assert len(calls) == 2
    assert obs.screenshot_path and pathlib.Path(obs.screenshot_path).exists()


async def test_demo_target_click_at_known_coordinates(controller):
    await controller.goto(DEMO.as_uri())
    box = await controller.page.locator("#start").bounding_box()
    assert box is not None
    assert (box["x"], box["y"], box["width"], box["height"]) == (100, 100, 160, 60)
    assert (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2) == (180, 130)
    await controller.page.mouse.click(180, 130)
    state = await controller.page.evaluate("window.__demoState")
    assert state["clicks"] == 1


async def test_demo_log_updates_on_click(controller):
    await controller.goto(DEMO.as_uri())
    await controller.page.click("#start")
    log = await controller.page.inner_text("#log")
    assert log == "clicks=1"


async def test_demo_records_last_key(controller):
    await controller.goto(DEMO.as_uri())
    await controller.page.keyboard.press("a")
    state = await controller.page.evaluate("window.__demoState")
    assert state["lastKey"] == "a"
