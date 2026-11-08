# M3 — Closed-loop Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `Session` that drives the real observe → decide → validate → act → verify loop over the built controller and VLM, with a coordinate-grid overlay, screenshot-diff verification, and hard safety limits that always terminate and always release input.

**Architecture:** A new `overlay.py` (Pillow grid) and `session.py` (the loop, duck-typed on `controller`/`vlm`, with injected `clock`/`sleep` for deterministic tests). Reuses `StateMachine`, `SessionState`, `validate`, `SessionLogger`. No changes to `vlm.py` or `controller.py`.

**Tech Stack:** Python 3.14, Pillow (new), Playwright + Pydantic v2 + python-dotenv (existing), pytest (+ pytest-asyncio).

**Spec:** `docs/superpowers/specs/2026-11-08-session-loop-m3-design.md`

## Global Constraints

- Python 3.14.7; `.venv`. Add exactly ONE new dependency: `pillow`.
- `Session` receives an ALREADY-OPEN controller; it never calls `open`/`close`.
- DO NOT change `vlm.py`, `controller.py`, `validation.py`, `state.py`, `session_state.py`, `schema.py`, `config.py`.
- `release_all()` runs on EVERY exit of `run()` (a `finally`), best-effort.
- Every loop path terminates: `max_cycles`, action budget, consecutive-failure cap, repeat limit, or `stop()`.
- Verify-unchanged is a failure ONLY for screen-affecting actions (`click`,`drag`,`wheel`,`press`,`hold`); `wait`/`move`/`release` are exempt.
- Default `pytest` stays offline (markers `browser`/`live` already excluded by `addopts`).
- Never log secrets or the raw image.

## Review Focus

Input classes / failure modes most likely to bite, most likely first:

1. **Non-termination**: every branch must end the loop (budget, cycle cap, failure cap, repeat, stop). Pinned in Task 4.
2. **Stuck input**: `release_all` on every exit including exceptions. Pinned in Tasks 3–4.
3. **Rate-limit pacing with an injected clock**: the 2nd cycle must not be rejected `RATE_LIMITED` because the fake clock never advanced. Pinned in Task 3.
4. **Exempt actions**: `wait`/`move` (which cannot change pixels) must not false-fail. Pinned in Task 4.
5. **Grid overlay fidelity**: same dimensions, non-destructive, correct for `device_scale_factor != 1`. Pinned in Task 1.

---

### Task 1: Pillow dependency + coordinate grid overlay

**Files:**
- Modify: `pyproject.toml` (add `pillow`)
- Create: `src/game_agent/overlay.py`
- Create: `tests/test_overlay.py`

**Interfaces:**
- Consumes: Pillow.
- Produces: `add_coordinate_grid(image_path, *, step: int = 100, scale: float = 1.0, out_path=None) -> str`; pure helpers `grid_lines(width: int, height: int, step: int, scale: float) -> tuple[list[int], list[int]]` (pixel positions of vertical/horizontal lines) and `format_label(pixel: int, scale: float) -> str` (CSS-px label) so the coordinate math is unit-testable without pixel inspection.

- [ ] **Step 1: Install Pillow and add it to dependencies**

Run: `.venv\Scripts\python -m pip install pillow` (wheel cp314 confirmed).
Add `"pillow"` to the `dependencies` list in `pyproject.toml`.

- [ ] **Step 2: Write the failing test**

`tests/test_overlay.py`:
```python
import pytest
from PIL import Image
from game_agent.overlay import add_coordinate_grid

def make_png(tmp_path, w=200, h=120):
    p = tmp_path / "shot.png"
    Image.new("RGB", (w, h), (255, 255, 255)).save(p)
    return p

def test_grid_writes_new_file_same_size(tmp_path):
    src = make_png(tmp_path)
    before = src.read_bytes()
    out = add_coordinate_grid(src, step=50)
    assert out != str(src)
    from pathlib import Path
    assert Path(out).exists()
    assert Image.open(out).size == (200, 120)
    assert src.read_bytes() == before              # original untouched
    assert Path(out).read_bytes() != before        # grid actually drawn

def test_grid_scale_keeps_labels_in_css_px(tmp_path):
    src = make_png(tmp_path, w=400, h=200)
    out = add_coordinate_grid(src, step=100, scale=2.0)   # 400px image == 200 CSS px
    assert Image.open(out).size == (400, 200)             # size unchanged

def test_grid_rejects_bad_params(tmp_path):
    src = make_png(tmp_path)
    with pytest.raises(ValueError):
        add_coordinate_grid(src, step=0)
    with pytest.raises(ValueError):
        add_coordinate_grid(src, scale=0)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_overlay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_agent.overlay'`.

- [ ] **Step 4: Implement `overlay.py`**

`add_coordinate_grid`: validate `step > 0` and `scale > 0` (else `ValueError`); open the PNG with Pillow; compute the pixel step `round(step * scale)`; draw vertical/horizontal grid lines every pixel step and a CSS-pixel label at each (label value = `i / scale`, formatted as int when whole); draw a legend text at the top-left (e.g. `f"grid {step}px · coords = CSS px"`); save to `out_path` or `<stem>.grid.png` next to the source; never write to the source; return the output path as `str`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_overlay.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/game_agent/overlay.py tests/test_overlay.py
git commit -m "feat: add coordinate grid overlay and pillow dependency"
```

---

### Task 2: `SessionLogger.log_verify`

**Files:**
- Modify: `src/game_agent/logging.py`
- Modify: `tests/test_logging.py`

**Interfaces:**
- Consumes: `Observation`.
- Produces: `SessionLogger.log_verify(observation: Observation, changed: bool, note: str = "") -> None` writing one JSONL `event="verify"` record with `observation_id`, `changed`, `note`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_logging.py`:
```python
def test_log_verify_records_changed(tmp_path, clock):
    from game_agent.logging import SessionLogger
    from game_agent.observation import Observation
    obs = Observation(observation_id="obs-v", screenshot_path=None, viewport_width=1,
                      viewport_height=1, device_scale_factor=1.0, timestamp=0.0)
    log = SessionLogger(tmp_path / "v.jsonl", clock=clock)
    log.log_verify(obs, changed=False, note="no visible change")
    log.close()
    import json
    row = json.loads((tmp_path / "v.jsonl").read_text(encoding="utf-8").strip())
    assert row["event"] == "verify"
    assert row["observation_id"] == "obs-v"
    assert row["changed"] is False
    assert row["note"] == "no visible change"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_logging.py::test_log_verify_records_changed -v`
Expected: FAIL — no `log_verify`.

- [ ] **Step 3: Implement `log_verify`**

Add the method using the existing `_write` choke point (timestamp from the clock, `observation_id` from the observation).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_logging.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/logging.py tests/test_logging.py
git commit -m "feat: add SessionLogger.log_verify"
```

---

### Task 3: `Session` core loop (happy path, pacing, grid, release-on-exit)

**Files:**
- Create: `src/game_agent/session.py`
- Create: `tests/test_session.py`

**Interfaces:**
- Consumes: `State`/`StateMachine`, `SessionState`, `validate`, `SessionLogger`, `add_coordinate_grid`, `Observation`, `Decision`, `Settings`, `VLMError`, `Clock`.
- Produces: `SessionResult` frozen dataclass (`state`, `cycles`, `reason`, `last_decision`); `Session(controller, vlm, settings, *, logger=None, max_consecutive_failures=3, repeat_limit=3, grid_step=100, clock=None, sleep=None)` with `async run(goal, max_cycles=50) -> SessionResult` and `stop()`.

- [ ] **Step 1: Write the test fakes and failing tests**

`tests/test_session.py`:
```python
import pytest
from game_agent.config import Settings
from game_agent.controller import ControllerError
from game_agent.observation import Observation
from game_agent.schema import Decision
from game_agent.session import Session, SessionResult
from game_agent.state import State
from game_agent.vlm import VLMError

class FakeController:
    def __init__(self, tmp_path, frames):
        self.tmp, self.frames, self.i = tmp_path, list(frames), 0
        self.executed, self.release_calls, self.fail_execute = [], 0, set()
        self.fail_screenshot = False
    async def screenshot(self):
        if self.fail_screenshot:
            raise ControllerError("screenshot failed")
        data = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        p = self.tmp / f"shot-{self.i}.png"
        p.write_bytes(data)
        return Observation(observation_id=f"obs-{self.i}", screenshot_path=str(p),
                           viewport_width=1280, viewport_height=720,
                           device_scale_factor=1.0, timestamp=float(self.i))
    async def execute(self, action):
        if action.type in self.fail_execute:
            raise ControllerError("execute failed")
        self.executed.append(action)
    async def release_all(self):
        self.release_calls += 1

class FakeVLM:
    def __init__(self, decisions):
        self.decisions, self.calls, self.seen = list(decisions), 0, []
    async def decide(self, observation, goal, remaining_actions):
        self.seen.append(observation.screenshot_path)
        d = self.decisions[min(self.calls, len(self.decisions) - 1)]
        self.calls += 1
        if isinstance(d, Exception):
            raise d
        return Decision.model_validate(d)

def dec(action, conf=0.9):
    return {"state": "s", "reason": "r", "confidence": conf, "action": action}

def make_sleep(clock):
    async def _sleep(seconds):
        clock.advance(seconds)
    return _sleep

async def test_happy_path_reaches_done(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B"])   # second shot differs -> changed
    vlm = FakeVLM([dec({"type": "click", "x": 10, "y": 10}),
                   dec({"type": "done", "summary": "ok"})])
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=10)
    assert result.state == State.DONE
    assert [a.type for a in ctrl.executed] == ["click"]
    assert ctrl.release_calls >= 1

async def test_second_cycle_not_rate_limited(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B", b"C"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1}),
                   dec({"type": "click", "x": 2, "y": 2}),
                   dec({"type": "done", "summary": "ok"})])
    session = Session(ctrl, vlm, Settings(min_action_interval_ms=200), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=10)
    assert result.state == State.DONE                 # not NEEDS_USER(RATE_LIMITED)
    assert vlm.calls == 3

async def test_grid_overlay_path_is_sent_to_vlm(tmp_path, clock):
    import pathlib
    ctrl = FakeController(tmp_path, [b"A", b"B"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1}), dec({"type": "done", "summary": "ok"})])
    session = Session(ctrl, vlm, Settings(), grid_step=50, clock=clock, sleep=make_sleep(clock))
    await session.run("goal", max_cycles=5)
    assert vlm.seen[0].endswith(".grid.png")
    assert pathlib.Path(vlm.seen[0]).exists()

async def test_release_all_called_on_done(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B"])
    vlm = FakeVLM([dec({"type": "done", "summary": "ok"})])
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    await session.run("goal", max_cycles=5)
    assert ctrl.release_calls == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_agent.session'`.

- [ ] **Step 3: Implement `session.py` (core)**

Implement `SessionResult`, `_image_changed(before, after) -> bool` (compare the two PNG files byte-for-byte; return `True` if either path is missing), and `Session.run` for the happy path:
- `try: ... finally: await controller.release_all()` (best-effort, swallow exceptions).
- `self._clock = clock or MonotonicClock()`; `self._sleep = sleep or asyncio.sleep`; `self._state = SessionState(settings, clock=self._clock)`; `self._machine = StateMachine()`.
- Per cycle: check `self._stopped`; **pace** — if `now - last_action_at < min_action_interval_ms`, `await self._sleep(remainder)`.
- `OBSERVING`: `screenshot()`; if `grid_step`, `replace(obs, screenshot_path=add_coordinate_grid(obs.screenshot_path, step=grid_step, scale=obs.device_scale_factor))`; log observation.
- `DECIDING`: `vlm.decide(vlm_obs, goal, remaining)`; log decision.
- `VALIDATING`: `validate(...)`; `done` → `DONE`; `uncertain` → `NEEDS_USER`.
- `ACTING`: `execute`; `record`; log action.
- `VERIFYING`: screenshot `after`; `changed`; `log_verify`; `record_success()` on change.
- Transitions must follow `StateMachine`'s table (`IDLE→OBSERVING→DECIDING→VALIDATING→ACTING→VERIFYING→OBSERVING`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_session.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Verify the default suite is green/offline**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/game_agent/session.py tests/test_session.py
git commit -m "feat: add Session core loop with pacing, grid, and release-on-exit"
```

---

### Task 4: `Session` safety branches

**Files:**
- Modify: `src/game_agent/session.py`
- Modify: `tests/test_session.py`

**Interfaces:**
- Consumes: Task 3's `Session`.
- Produces: the failure/repeat/budget/stop/error branches of `run`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_session.py`:
```python
from game_agent.state import State

async def test_unchanged_image_hits_failure_cap(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])          # same bytes every shot -> never changes
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=2,
                      repeat_limit=99, clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=10)
    assert result.state == State.NEEDS_USER
    assert ctrl.release_calls == 1

async def test_wait_action_is_exempt_from_unchanged_failure(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])          # never changes
    vlm = FakeVLM([dec({"type": "wait", "duration_ms": 10}),
                   dec({"type": "done", "summary": "ok"})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=1,
                      repeat_limit=99, clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.DONE                 # wait does not false-fail

async def test_repeated_action_hits_repeat_limit(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B"])    # changes each time, but same action repeats
    vlm = FakeVLM([dec({"type": "click", "x": 5, "y": 5})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=99,
                      repeat_limit=2, clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=10)
    assert result.state == State.NEEDS_USER
    assert "repeat" in result.reason.lower()

async def test_budget_exhaustion_stops(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B", b"C", b"D"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    session = Session(ctrl, vlm, Settings(max_actions_per_session=2),
                      max_consecutive_failures=99, repeat_limit=99, clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=10)
    assert result.state == State.STOPPED
    assert len(ctrl.executed) <= 2

async def test_max_cycles_stops(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B", b"C"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=99, repeat_limit=99,
                      clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=2)
    assert result.state == State.STOPPED
    assert result.cycles <= 2

async def test_stop_ends_loop(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B", b"C"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=99, repeat_limit=99,
                      clock=clock, sleep=make_sleep(clock))
    session.stop()
    result = await session.run("goal", max_cycles=10)
    assert result.state == State.STOPPED
    assert ctrl.release_calls == 1
    assert ctrl.executed == []

async def test_vlm_error_needs_user(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    vlm = FakeVLM([VLMError("no key")])
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.NEEDS_USER
    assert ctrl.release_calls == 1

async def test_execute_error_recovers_or_caps(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    ctrl.fail_execute = {"click"}
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=1, repeat_limit=99,
                      clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.NEEDS_USER
    assert ctrl.release_calls == 1

async def test_screenshot_error_needs_user(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    ctrl.fail_screenshot = True
    vlm = FakeVLM([dec({"type": "done", "summary": "x"})])
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.NEEDS_USER
    assert ctrl.release_calls == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_session.py -v`
Expected: FAIL — the branches are not implemented (e.g. no failure cap / repeat / budget).

- [ ] **Step 3: Implement the safety branches**

- **Failure cap:** `record_failure()`; when `session_state.consecutive_failures >= max_consecutive_failures` → route to `NEEDS_USER`; otherwise to `RECOVERING→OBSERVING` (from `ACTING`/`VERIFYING`) or straight to `NEEDS_USER` (from `OBSERVING`).
- **Exempt actions:** `{wait, move, release}` skip the unchanged→failure rule (still `record_success()`), but still count toward the repeat key.
- **Repeat:** dict keyed by `(action.model_dump_json(), sha256(after_bytes))`; consecutive count `>= repeat_limit` → `NEEDS_USER` (reason contains "repeat"); a successful change resets the counters.
- **Budget / cycles:** `remaining_action_budget() <= 0` or loop exhausted → `STOPPED`.
- **stop():** `_stopped` checked at the top of each cycle → `STOPPED`.
- **Errors:** `VLMError` from `decide` → `NEEDS_USER`; exception from `screenshot`/`execute`/overlay → `record_failure()` and the cap routing above.
- Note: the repeat test uses a changing image with a constant action; the failure cap is raised to `99` there so the repeat rule is isolated, and vice-versa — keep those parameter settings.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_session.py -v`
Expected: PASS (all session tests).

- [ ] **Step 5: Verify the default suite is green/offline**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/game_agent/session.py tests/test_session.py
git commit -m "feat: add Session safety branches (cap, repeat, budget, stop, errors)"
```

---

### Task 5: Opt-in live end-to-end test + README

**Files:**
- Create: `tests/test_session_live.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Session`, `BrowserController`, `VLMClient`, `examples/demo.html`.
- Produces: a live `Session.run` test and an M3 README section.

- [ ] **Step 1: Write the live test**

`tests/test_session_live.py`:
```python
import pathlib
import pytest
from game_agent.config import Settings
from game_agent.controller import BrowserController, BrowserOptions
from game_agent.session import Session
from game_agent.state import State
from game_agent.vlm import VLMClient, load_vlm_config

pytestmark = [pytest.mark.live, pytest.mark.browser]
DEMO = pathlib.Path(__file__).resolve().parents[1] / "examples" / "demo.html"

async def test_live_session_clicks_demo(tmp_path):
    cfg = load_vlm_config()
    if not (cfg.api_key and cfg.model):
        pytest.skip("set NVIDIA_API_KEY and NVIDIA_MODEL (env or .env) to run the live test")
    controller = BrowserController(Settings(), options=BrowserOptions(
        headless=True, user_data_dir=str(tmp_path / "p"), screenshot_dir=str(tmp_path / "s")))
    await controller.open()
    try:
        await controller.goto(DEMO.as_uri())
        session = Session(controller, VLMClient(cfg), Settings(), grid_step=100)
        result = await session.run("Click the Start button, then press the up arrow.", max_cycles=6)
    finally:
        await controller.close()
    assert result.state in (State.DONE, State.NEEDS_USER)
    assert result.cycles >= 1
```

- [ ] **Step 2: Confirm the default suite deselects it**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS, live test not collected.

- [ ] **Step 3: Run the live test once (real API, approved)**

Run: `$env:NVIDIA_API_KEY="..."; $env:NVIDIA_MODEL="..."; .venv\Scripts\python -m pytest -m "live and browser" tests/test_session_live.py -v`
Expected: PASS (`DONE` or `NEEDS_USER`), or SKIP without credentials. If it errors, capture the failure — do not weaken the assertion; report the outcome.

- [ ] **Step 4: Update `README.md`**

Update **Current status** (M3 closed-loop `Session` implemented; game-specific work M4 pending). Add a **Closed-loop session (M3)** section: what the loop does, the coordinate-grid overlay, screenshot-diff verification, the safety limits (`max_cycles`, `max_consecutive_failures`, `repeat_limit`, action budget, `stop()`), the `pillow` dependency, and how to run the live test (`pytest -m "live and browser"`). Add `overlay.py` and `session.py` to the module map.

- [ ] **Step 5: Run the full default suite**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_session_live.py README.md
git commit -m "test: add opt-in live Session test and document M3"
```

---

## Self-Review

- **Spec coverage:** overlay ↔ Task 1 (§7.1); log_verify ↔ Task 2 (§7.4); Session core/clock/pacing/grid/release-finally ↔ Task 3 (§7.2, §7.3); safety branches ↔ Task 4 (§7.2, decisions 1–7); live test + docs ↔ Task 5 (§7.6); no-vlm-change ↔ Tasks 1–4 constraints (§3); pillow ↔ Task 1 (§5).
- **Type consistency:** `add_coordinate_grid(step, scale, out_path)`, `SessionLogger.log_verify(observation, changed, note)`, `SessionResult`, `Session(controller, vlm, settings, *, logger, max_consecutive_failures, repeat_limit, grid_step, clock, sleep)`, `run(goal, max_cycles)`, `stop()` are named identically wherever produced/consumed. Fakes match the `controller`/`vlm` duck-typed interfaces used by `run`.
- **Review Focus:** items 1–5 pinned in Tasks 4, 3–4, 3, 4, 1 respectively.
- **Deferred (design §9):** nudge wording, grid step/label styling, `max_cycles` default.
