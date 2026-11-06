"""Browser tests for the real Playwright-backed controller.

These are opt-in: the ``browser`` marker keeps them out of the default
(offline) suite.  Run them with::

    .venv\\Scripts\\python -m pytest -m browser tests/test_browser_controller.py -v

If the Chromium executable is not installed yet, the tests skip with a clear
message instead of failing, so a fresh checkout is never red.
"""

import pathlib
import struct
import time

import pytest
from playwright.async_api import Error as PlaywrightError

from game_agent.config import Settings
from game_agent.controller import (
    ActionNotAllowed,
    BrowserController,
    BrowserOptions,
)
from game_agent.observation import Observation
from game_agent.schema import (
    ClickAction,
    DragAction,
    HoldAction,
    MoveAction,
    PressAction,
    ReleaseAction,
    WaitAction,
    WheelAction,
)

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


async def test_click_hits_demo_target(controller):
    await controller.goto(DEMO.as_uri())
    await controller.execute(ClickAction(type="click", x=180, y=130))
    state = await controller.page.evaluate("window.__demoState")
    assert state["clicks"] == 1


async def test_press_translates_agent_key(controller):
    await controller.goto(DEMO.as_uri())
    await controller.execute(PressAction(type="press", key="up"))
    state = await controller.page.evaluate("window.__demoState")
    assert state["lastKey"] == "ArrowUp"


async def test_hold_leaves_no_stuck_input(controller):
    await controller.goto(DEMO.as_uri())
    await controller.execute(HoldAction(type="hold", key="w", duration_ms=50))
    assert controller.held_keys == frozenset()


async def test_release_all_clears_after_failed_up(controller, monkeypatch):
    await controller.goto(DEMO.as_uri())
    calls = {"n": 0}
    real = controller._release_key

    async def flaky(key):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        await real(key)

    monkeypatch.setattr(controller, "_release_key", flaky)
    with pytest.raises(Exception):
        await controller.execute(HoldAction(type="hold", key="w", duration_ms=50))
    assert controller.held_keys == frozenset({"w"})   # still tracked
    await controller.release_all()
    assert controller.held_keys == frozenset()        # cleared


async def test_execute_rejects_done(controller):
    from game_agent.schema import DoneAction
    with pytest.raises(ActionNotAllowed):
        await controller.execute(DoneAction(type="done", summary="x"))


async def test_release_all_survives_one_failing_release(controller, monkeypatch):
    await controller.goto(DEMO.as_uri())
    await controller._hold_key("w")
    await controller._hold_key("a")
    real = controller._release_key

    async def flaky(key):
        if key == "w":
            raise RuntimeError("boom")
        await real(key)

    monkeypatch.setattr(controller, "_release_key", flaky)
    await controller.release_all()
    assert controller.held_keys == frozenset({"w"})   # the failed one stays
    assert controller.held_buttons == frozenset()


async def test_hold_button_leaves_no_stuck_button(controller):
    await controller.goto(DEMO.as_uri())
    await controller.execute(HoldAction(type="hold", button="left", duration_ms=50))
    assert controller.held_buttons == frozenset()


async def test_release_is_idempotent_when_nothing_is_held(controller):
    await controller.goto(DEMO.as_uri())
    await controller.execute(ReleaseAction(type="release", key="w"))
    await controller.execute(ReleaseAction(type="release", button="left"))
    assert controller.held_keys == frozenset()
    assert controller.held_buttons == frozenset()


async def test_move_moves_the_pointer(controller):
    await controller.goto(DEMO.as_uri())
    await controller.page.evaluate(
        "() => { window.__moves = [];"
        " document.addEventListener('mousemove',"
        " e => window.__moves.push([e.clientX, e.clientY])); }"
    )
    await controller.execute(MoveAction(type="move", x=123, y=45))
    # Pointer moves reach the page asynchronously; wait for the event.
    await controller.page.wait_for_function("() => window.__moves.length > 0")
    moves = await controller.page.evaluate("window.__moves")
    assert moves[-1] == [123, 45]


async def test_drag_presses_and_releases_through_the_ledger(controller):
    await controller.goto(DEMO.as_uri())
    await controller.page.evaluate(
        "() => { window.__events = [];"
        " document.addEventListener('mousedown', () => window.__events.push(['down']));"
        " document.addEventListener('mouseup', () => window.__events.push(['up']));"
        " document.addEventListener('mousemove',"
        " e => window.__events.push(['move', e.clientX, e.clientY])); }"
    )
    await controller.execute(
        DragAction(type="drag", start_x=100, start_y=100,
                   end_x=300, end_y=200, duration_ms=50)
    )
    # Pointer moves reach the page asynchronously; wait for the release.
    await controller.page.wait_for_function(
        "() => window.__events.some(e => e[0] === 'up')"
    )
    events = await controller.page.evaluate("window.__events")
    moves = [e for e in events if e[0] == "move"]
    assert events[0] == ["move", 100, 100]   # the pointer travels to the start
    assert ["down"] in events
    assert events[-1] == ["up"]
    assert len(moves) >= 2            # interpolated, not a single jump
    assert moves[-1] == ["move", 300, 200]
    assert controller.held_buttons == frozenset()


async def test_wheel_scrolls_at_the_tracked_cursor(controller):
    await controller.goto(DEMO.as_uri())
    await controller.page.evaluate("document.body.style.height = '3000px'")
    await controller.page.evaluate(
        "() => { window.__wheel = [];"
        " document.addEventListener('wheel',"
        " e => window.__wheel.push([e.deltaX, e.deltaY])); }"
    )
    await controller.execute(MoveAction(type="move", x=640, y=360))
    await controller.execute(WheelAction(type="wheel", delta_y=300))
    # The wheel event reaches the page asynchronously; wait for it.
    await controller.page.wait_for_function("() => window.__wheel.length > 0")
    wheel = await controller.page.evaluate("window.__wheel")
    assert wheel[-1] == [0, 300]
    assert await controller.page.evaluate("window.scrollY") > 0


async def test_wait_action_waits_for_the_requested_duration(controller):
    await controller.goto(DEMO.as_uri())
    started = time.monotonic()
    await controller.execute(WaitAction(type="wait", duration_ms=50))
    assert time.monotonic() - started >= 0.05
