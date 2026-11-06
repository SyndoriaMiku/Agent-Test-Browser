# M1 — Browser Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the `BrowserController` stub into a real Playwright-backed executor that opens a dedicated browser, screenshots at a known viewport, and performs mouse/keyboard actions with a safe held-input guarantee — verified against a local demo page.

**Architecture:** A pure translation/policy layer (key names, action pre-checks) testable offline, plus a Playwright async controller that owns browser lifecycle, a held-input ledger, and best-effort `release_all`. Browser tests are opt-in (`browser` marker) so the default suite stays offline.

**Tech Stack:** Python 3.14, Playwright (async API, bundled Chromium), pytest (+ pytest-asyncio), Pydantic v2 (existing).

**Spec:** `docs/superpowers/specs/2026-11-06-browser-controller-m1-design.md`

## Global Constraints

- Python interpreter: machine `python` = **3.14.7**; env at `.venv`. Only dependency used: `playwright` (already declared).
- Default `pytest` MUST stay offline: register the `browser` marker and set `addopts = "-m \"not browser\""` so browser tests never run by default.
- Browser: bundled Chromium via `playwright install chromium`; dedicated `user_data_dir` (`.browser-profile/`, git-ignored); NEVER the user's Chrome/Edge profile.
- Viewport `1280x720` CSS px, `device_scale_factor` from `Settings` (the single source of truth — `BrowserOptions` must not redeclare it).
- Coordinates are CSS pixels; the layout viewport never changes mid-session.
- Reuse `Observation`, `Action`, `Settings`, `Clock`; never write secrets/cookies/tokens to a log.
- `hold` is atomic (down → wait → up in `try/finally`); `release` is idempotent; `release_all` is best-effort and leaves no stuck input.

## Review Focus

Input classes / failure modes most likely to bite, most likely first:

1. **Stuck input after a failed/cancelled action**: a `hold` whose `up` fails must leave the key tracked so `release_all()` clears it. Pinned in Task 4.
2. **Key-name vocabulary mismatch**: `Settings.allowed_keys` uses `up`/`space`/`enter`, but Playwright needs `ArrowUp`/`Space`/`Enter` — `press` must translate. Pinned in Task 1 (unit) and Task 4 (end-to-end).
3. **Default suite must not open a browser**: the marker + `addopts` must keep `pytest` offline. Pinned in Task 1.
4. **Coordinate/viewport consistency**: the click must land on the demo target at the screenshot's coordinate system. Pinned in Tasks 3–4.
5. **Missing browser executable**: opt-in tests must skip with a clear message, not fail, so a fresh checkout is never red. Pinned in Task 3.

---

### Task 1: Key translation + browser-test wiring

**Files:**
- Create: `src/game_agent/keys.py`
- Create: `tests/test_keys.py`
- Modify: `pyproject.toml` (register `browser` marker; `addopts`)

**Interfaces:**
- Consumes: nothing.
- Produces: `to_playwright_key(name: str) -> str` — maps the agent vocabulary to Playwright key names, passes single characters and unknown names through unchanged.

- [ ] **Step 1: Write the failing test**

`tests/test_keys.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_keys.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_agent.keys'`.

- [ ] **Step 3: Implement `keys.py`**

A single module-level `dict[str, str]` `_KEY_MAP` for the nine agent names and `to_playwright_key(name: str) -> str` returning `_KEY_MAP.get(name, name)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_keys.py -v`
Expected: PASS (11 cases).

- [ ] **Step 5: Register the `browser` marker and keep the default suite offline**

In `pyproject.toml` under `[tool.pytest.ini_options]`, add:
```toml
addopts = "-m \"not browser\""
markers = ["browser: real-browser tests (opt-in; run with -m browser)"]
```

- [ ] **Step 6: Verify default suite still offline**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS — all existing tests, no browser opened.

- [ ] **Step 7: Commit**

```bash
git add src/game_agent/keys.py tests/test_keys.py pyproject.toml
git commit -m "feat: add key-name translation and browser test marker"
```

---

### Task 2: Browser options, exceptions, and the action policy check

**Files:**
- Modify: `src/game_agent/controller.py`
- Create: `tests/test_controller_policy.py`

**Interfaces:**
- Consumes: `Settings` (config), `Action` (schema).
- Produces: `BrowserOptions` frozen dataclass (`headless: bool = False`, `user_data_dir: str = ".browser-profile"`, `screenshot_dir: str = "screenshots"`, `browser_channel: str | None = None`); `ControllerError(Exception)`, `ActionNotAllowed(ControllerError)`, `ActionExecutionError(ControllerError)`; module-level `assert_executable(action: Action, settings: Settings) -> None` raising `ActionNotAllowed`.

- [ ] **Step 1: Write the failing tests**

`tests/test_controller_policy.py`:
```python
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
    with pytest.raises(Exception):
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_controller_policy.py -v`
Expected: FAIL — `ImportError` (names not defined in `controller.py`).

- [ ] **Step 3: Implement options, exceptions, and `assert_executable` in `controller.py`**

Keep the existing `BrowserController` class below the new code. `assert_executable` order: `done`/`uncertain` → `ActionNotAllowed`; coordinate bounds for `click`/`move`/`drag` (closed interval `0 <= v <= viewport_*`); key allowlist for `press`/`hold`/`release` (compare the agent name against `settings.allowed_keys`); button allowlist for `click`/`drag`/`hold`/`release`; duration caps (`wait` ≤ `max_wait_ms`, `hold` ≤ `max_hold_ms`, `drag` ≤ `max_wait_ms`). It is a pure function (no I/O).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_controller_policy.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/controller.py tests/test_controller_policy.py
git commit -m "feat: add browser options, controller errors, and action policy check"
```

---

### Task 3: Browser lifecycle + screenshot + demo page

**Files:**
- Create: `examples/demo.html`
- Modify: `src/game_agent/controller.py` (real `open`/`goto`/`page`/`screenshot`/`close`/`release_all`)
- Create: `tests/test_browser_controller.py`
- Modify: `.gitignore` (add `.browser-profile/`)

**Interfaces:**
- Consumes: `BrowserOptions`, `Settings`, `Observation`, `Clock`, `MonotonicClock` (session_state), `to_playwright_key` (Task 1).
- Produces: `BrowserController(settings, options=None, clock=None)` with async `open()`, `goto(url)`, `screenshot() -> Observation`, `release_all()`, `close()`, and property `page`.

- [ ] **Step 1: Install the browser (one-time setup)**

Run: `.venv\Scripts\python -m playwright install chromium`
Expected: Chromium downloads and installs (approved, ~150MB).

- [ ] **Step 2: Create the demo page**

`examples/demo.html` — a static page (no network) with `#start` (absolute at left:100px;top:100px;width:160px;height:60px, so its center is viewport `(180,130)`), a `#log` element, `window.__demoState = {clicks:0, lastKey:null}`, a click listener on `#start` that increments `clicks` and writes to `#log`, and a `keydown` listener that records `e.key`.

- [ ] **Step 3: Add `.browser-profile/` to `.gitignore`**

Append `.browser-profile/` to `.gitignore`.

- [ ] **Step 4: Write the failing browser test**

`tests/test_browser_controller.py`:
```python
import pathlib
import pytest
from game_agent.config import Settings
from game_agent.controller import BrowserController, BrowserOptions
from game_agent.observation import Observation

pytestmark = pytest.mark.browser
DEMO = pathlib.Path(__file__).resolve().parents[1] / "examples" / "demo.html"

@pytest.fixture
async def controller(tmp_path):
    opts = BrowserOptions(headless=True, user_data_dir=str(tmp_path / "profile"),
                          screenshot_dir=str(tmp_path / "shots"))
    c = BrowserController(Settings(), options=opts)
    await c.open()
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
```

- [ ] **Step 5: Run the browser test to verify it fails**

Run: `.venv\Scripts\python -m pytest -m browser tests/test_browser_controller.py -v`
Expected: FAIL — `BrowserController.open` still raises `NotImplementedError`.

- [ ] **Step 6: Implement lifecycle in `controller.py`**

`open()` uses `playwright.async_api` → `chromium.launch_persistent_context(user_data_dir, headless=…, viewport={"width":settings.viewport_width,"height":settings.viewport_height}, device_scale_factor=settings.device_scale_factor, channel=options.browser_channel)`, stores context + first page. `goto` calls `page.goto(url, wait_until="load")`. `page` property returns the page. `screenshot` saves to `<screenshot_dir>/<session_id>/<observation_id>.png` and returns an `Observation` (viewport from `settings`, timestamp from the injected clock). `release_all` iterates the (currently empty) ledger best-effort. `close` runs `release_all` then closes context/playwright in `try/finally`. Update the stub docstrings to reflect the real behavior.

- [ ] **Step 7: Run the browser test to verify it passes**

Run: `.venv\Scripts\python -m pytest -m browser tests/test_browser_controller.py -v`
Expected: PASS (2 tests). If the browser executable is missing, the tests must `skip` with a clear message instead of failing (wrap the fixture's `open()` and `pytest.skip` on the executable-missing error).

- [ ] **Step 8: Verify the default suite is still offline and green**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS — no browser tests collected.

- [ ] **Step 9: Commit**

```bash
git add examples/demo.html tests/test_browser_controller.py src/game_agent/controller.py .gitignore
git commit -m "feat: implement browser lifecycle, navigation, and screenshots"
```

---

### Task 4: Action execution + held-input ledger

**Files:**
- Modify: `src/game_agent/controller.py`
- Modify: `tests/test_browser_controller.py`

**Interfaces:**
- Consumes: `assert_executable`, `to_playwright_key`, everything from Task 3.
- Produces: `BrowserController.execute(action) -> None`; private ledger helpers `_hold_key(key)`, `_release_key(key)`, `_hold_button(button)`, `_release_button(button)`, and readonly `held_keys` / `held_buttons` for tests.

- [ ] **Step 1: Write the failing browser tests**

Add to `tests/test_browser_controller.py`:
```python
from game_agent.schema import ClickAction, HoldAction, PressAction, WaitAction
from game_agent.controller import ActionNotAllowed

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
```

- [ ] **Step 2: Run the browser tests to verify they fail**

Run: `.venv\Scripts\python -m pytest -m browser tests/test_browser_controller.py -v`
Expected: FAIL — `execute` still raises `NotImplementedError` / helpers missing.

- [ ] **Step 3: Implement `execute` and the ledger helpers**

`execute` calls `assert_executable(action, self._settings)` first, then dispatches: `click`→`mouse.click`; `move`→`mouse.move` (update `_cursor`); `press`→`keyboard.press(to_playwright_key(key))`; `hold`→`_hold_key`/`_hold_button`, `asyncio.sleep(duration_ms/1000)`, then release in `try/finally`; `release`→idempotent up; `drag`→`mouse.move(start)`, down, `mouse.move(end, steps=…)`, up (tracked); `wheel`→`mouse.move(self._cursor)` then `mouse.wheel`; `wait`→`asyncio.sleep`. Ledger helpers add the input to the set BEFORE awaiting the browser `down`, and remove it AFTER a successful `up`; `release_all` calls the release helpers for every tracked input, swallowing exceptions so one failure cannot block the rest. Expose `held_keys`/`held_buttons` as read-only properties returning frozensets.

- [ ] **Step 4: Run the browser tests to verify they pass**

Run: `.venv\Scripts\python -m pytest -m browser tests/test_browser_controller.py -v`
Expected: PASS (all browser tests).

- [ ] **Step 5: Verify the default suite is still green and offline**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS — no browser tests collected.

- [ ] **Step 6: Commit**

```bash
git add src/game_agent/controller.py tests/test_browser_controller.py
git commit -m "feat: implement action execution with held-input ledger"
```

---

### Task 5: README update for M1

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: an accurate README describing M1.

- [ ] **Step 1: Update `README.md`**

Update the **Current status** section to say the browser controller is now implemented (M1) while VLM (M2) and the loop (M3) are still pending. Add a **Browser PoC** section: install/run prerequisites (`playwright install chromium`), how to run the opt-in tests (`.venv\Scripts\python -m pytest -m browser`), and a note that the default suite stays offline. Add `keys.py` to the module map and update the `controller.py` row to describe the real behavior (lifecycle, `goto`, screenshot, execute, held-input ledger, `release_all`).

- [ ] **Step 2: Sanity-check the README against the code**

Confirm every module named exists under `src/game_agent/` and every command matches `pyproject.toml`.

- [ ] **Step 3: Run the full default suite**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS — all offline tests green.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: describe M1 browser controller in README"
```

---

## Self-Review

- **Spec coverage:** key translation ↔ Task 1 (§7.6); options/exceptions/policy ↔ Task 2 (§7.1, §7.5); lifecycle/goto/screenshot/demo ↔ Task 3 (§7.2, §7.3, §7.7); execute/ledger/hold/drag/wheel/release_all ↔ Task 4 (§7.4); offline default suite ↔ Task 1 + every task's "default suite" step (§7.8); README ↔ Task 5 (§7.8/§2). Constructor signature `(settings, options, clock)` ↔ Task 3.
- **Type consistency:** `to_playwright_key`, `BrowserOptions` fields, `assert_executable`, `ControllerError`/`ActionNotAllowed`/`ActionExecutionError`, `BrowserController.execute`, `held_keys`/`held_buttons` are named identically wherever they are produced and consumed.
- **Review Focus:** items 1–5 pinned in Tasks 4, 1+4, 1, 3–4, 3 respectively.
- **Deferred (design §9):** exact demo coordinates (fixed to `(180,130)` here), `session_id` generation, drag `steps`.
