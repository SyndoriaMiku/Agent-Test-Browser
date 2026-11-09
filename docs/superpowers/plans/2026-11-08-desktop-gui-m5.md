# M5 — Desktop GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Tkinter desktop GUI to drive and watch the agent — URL/goal/viewport inputs, Start/Pause/Resume/Stop, live grid-screenshot preview, live interaction log, status, and model token/cost — with the non-GUI logic unit-tested offline.

**Architecture:** Small additive core changes (Session pause/resume + read-only accessors, VLM usage/model, a `log_model` event, action-on-failure logging), then a headless `gui_model.py` (pure display logic), a `gui_runner.py` (background thread running `Session`, emitting events on a queue), and a thin Tk `gui.py` view.

**Tech Stack:** Python 3.14, stdlib `tkinter` + `queue` + `threading`, Pillow (`ImageTk`), existing Playwright/Pydantic/pytest.

**Spec:** `docs/superpowers/specs/2026-11-08-desktop-gui-m5-design.md`

## Global Constraints

- GUI uses stdlib `tkinter` + Pillow only; NO new heavy dependency.
- Core changes are additive only: `Session.pause()/resume()` + `state`/`cycles`/`remaining_actions`; `VLMClient.model`/`last_usage`; `SessionLogger.log_model`; `Session` calls `log_action` on execute failure. Do not break existing APIs/tests.
- The Tk view is thin; all non-GUI logic (gui_model, gui_runner) is offline-testable. Any test needing a display is marker `gui` and excluded by default (`addopts`).
- Never display or log the API key.
- `NEEDS_USER` is terminal (no mid-run guidance); pause takes effect only between cycles; `stop` beats `pause`.

## Review Focus

Most likely to bite, first:

1. **UI thread never blocks**: the agent runs off the Tk thread; all cross-thread data goes through the queue. Pinned in Task 4/5.
2. **Thread/loop/browser lifecycle**: Stop closes the browser only after `run()` returns; a normal end leaves it open; window close stops + closes. Pinned in Task 4.
3. **Pause determinism**: the pause gate uses the injected `_sleep`, and `stop` wins over `pause`. Pinned in Task 2.
4. **Usage accuracy**: `last_usage` sums repair calls (no undercount) and is `None` when absent. Pinned in Task 1.
5. **Preview path**: grid image used when present, raw otherwise, `None`-safe. Pinned in Task 3.

---

### Task 1: Core — VLM usage/model, `log_model`, Session emits model event

**Files:**
- Modify: `src/game_agent/vlm.py`
- Modify: `src/game_agent/logging.py`
- Modify: `src/game_agent/session.py`
- Create: `tests/test_vlm_usage.py`
- Modify: `tests/test_logging.py`

**Interfaces:**
- Produces: `VLMClient.model: str | None` and `VLMClient.last_usage: dict | None` (instance attributes set in `decide`); `SessionLogger.log_model(observation, model: str, usage: dict | None) -> None`; Session logs a `model` event after a decision when the VLM exposes usage.

- [ ] **Step 1: Write failing tests**

`tests/test_vlm_usage.py`:
```python
import pytest
from game_agent.vlm import VLMClient, VLMConfig

def _resp(content, usage=None):
    r = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        r["usage"] = usage
    return r

VALID = '{"state":"s","reason":"r","confidence":0.9,"action":{"type":"wait","duration_ms":10}}'

async def test_usage_summed_across_repair(tmp_path):
    p = tmp_path / "s.png"; p.write_bytes(b"\x89PNG\r\n\x1a\n")
    seq = [_resp("garbage", {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}),
           _resp(VALID, {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14})]
    calls = {"i": 0}
    def fake(url, headers, payload, t):
        r = seq[calls["i"]]; calls["i"] += 1; return r
    from game_agent.observation import Observation
    obs = Observation("o", str(p), 10, 10, 1.0, 0.0)
    client = VLMClient(VLMConfig(model="m", api_key="k"), transport=fake)
    await client.decide(obs, "goal", 1)
    assert client.model == "m"
    assert client.last_usage == {"prompt_tokens": 21, "completion_tokens": 5, "total_tokens": 26}

async def test_usage_none_when_absent(tmp_path):
    p = tmp_path / "s.png"; p.write_bytes(b"\x89PNG\r\n\x1a\n")
    from game_agent.observation import Observation
    obs = Observation("o", str(p), 10, 10, 1.0, 0.0)
    client = VLMClient(VLMConfig(model="m", api_key="k"),
                       transport=lambda *a: _resp(VALID))
    await client.decide(obs, "goal", 1)
    assert client.last_usage is None
```

Add to `tests/test_logging.py`:
```python
def test_log_model_records_tokens(tmp_path, clock):
    from game_agent.logging import SessionLogger
    from game_agent.observation import Observation
    obs = Observation("obs-m", None, 1, 1, 1.0, 0.0)
    log = SessionLogger(tmp_path / "m.jsonl", clock=clock)
    log.log_model(obs, "some-model", {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12})
    log.close()
    import json
    row = json.loads((tmp_path / "m.jsonl").read_text(encoding="utf-8").strip())
    assert row["event"] == "model" and row["observation_id"] == "obs-m"
    assert row["model"] == "some-model" and row["prompt_tokens"] == 5 and row["completion_tokens"] == 7
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv\Scripts\python -m pytest tests/test_vlm_usage.py -v`
Expected: FAIL (`last_usage`/`model` missing).

- [ ] **Step 3: Implement**

- `VLMClient.__init__`: `self.model = None`, `self.last_usage = None`.
- In `decide`: at the start, reset `self.last_usage = None`; after resolving config set `self.model = config.model`; accumulate a usage dict across each response (`prompt_tokens`/`completion_tokens`/`total_tokens` summed), tolerant of a missing/short `usage`.
- `SessionLogger.log_model`: one `event="model"` record with `observation_id`, `model`, and the three token fields (read via `.get`).
- `Session`: after a successful `decide`, `usage = getattr(self._vlm, "last_usage", None)`; if truthy, call `log_model = getattr(self._logger, "log_model", None)` and if callable `log_model(obs, getattr(self._vlm, "model", None) or "", usage)`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python -m pytest tests/test_vlm_usage.py tests/test_logging.py -v` then `.venv\Scripts\python -m pytest -q`
Expected: PASS; default suite green/offline.

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/vlm.py src/game_agent/logging.py src/game_agent/session.py tests/test_vlm_usage.py tests/test_logging.py
git commit -m "feat: capture model usage and log a model event"
```

---

### Task 2: Core — `Session` pause/resume, accessors, action-on-failure logging

**Files:**
- Modify: `src/game_agent/session.py`
- Create: `tests/test_session_pause.py`

**Interfaces:**
- Produces: `Session.pause()`, `Session.resume()`, and read-only `Session.state -> State`, `Session.cycles -> int`, `Session.remaining_actions -> int`.

- [ ] **Step 1: Write the failing test**

`tests/test_session_pause.py`:
```python
import pytest
from game_agent.config import Settings
from game_agent.schema import Decision
from game_agent.session import Session
from game_agent.state import State

# reuse the FakeController/FakeVLM/dec/make_sleep helpers pattern from tests/test_session.py
from tests.test_session import FakeController, FakeVLM, dec, make_sleep  # type: ignore

async def test_pause_blocks_then_resume_continues(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B", b"C"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1}), dec({"type": "done", "summary": "ok"})])
    calls = {"n": 0}
    session_ref = {}
    async def resume_sleep(seconds):
        clock.advance(seconds)
        calls["n"] += 1
        if calls["n"] >= 2:            # gate entered >= twice while paused, then resume
            session_ref["s"].resume()
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=resume_sleep)
    session_ref["s"] = session
    session.pause()
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.DONE       # paused, then resumed, then ran to DONE
    assert calls["n"] >= 2                  # the pause gate actually looped

async def test_stop_beats_pause(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    vlm = FakeVLM([dec({"type": "done", "summary": "x"})])
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    session.pause(); session.stop()
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.STOPPED and ctrl.executed == []

async def test_accessors(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1}), dec({"type": "done", "summary": "ok"})])
    session = Session(ctrl, vlm, Settings(max_actions_per_session=5), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert session.state == State.DONE and session.cycles == result.cycles
    assert session.remaining_actions == 5 - len(ctrl.executed)
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv\Scripts\python -m pytest tests/test_session_pause.py -v`
Expected: FAIL (`pause`/`resume`/accessors missing).

- [ ] **Step 3: Implement**

- `__init__`: `self._paused = False`; track `self._cycles = 0`.
- `pause()`: `self._paused = True`; `resume()`: `self._paused = False`.
- At the very top of each cycle, BEFORE the `_stopped` check: `while self._paused and not self._stopped: await self._sleep(0.1)`.
- Increment `self._cycles` each iteration; `remaining_actions` returns `self._state.remaining_action_budget()`.
- Properties: `state` → `self._machine.state`.
- On `execute` failure, also call `self._logger.log_action(obs, action, result)` (best-effort) before failure accounting.

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python -m pytest tests/test_session_pause.py tests/test_session.py -v` then `.venv\Scripts\python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/session.py tests/test_session_pause.py
git commit -m "feat: add Session pause/resume, accessors, and failure logging"
```

---

### Task 3: `gui_model.py` (pure display logic)

**Files:**
- Create: `src/game_agent/gui_model.py`
- Create: `tests/test_gui_model.py`

**Interfaces:**
- Produces: `format_event(event: dict) -> str`; `preview_path(screenshot_path: str | None) -> str | None`; `UsageTracker` with `add(usage)`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost_usd(in_per_1m: float | None, out_per_1m: float | None) -> float | None`.

- [ ] **Step 1: Write the failing test**

`tests/test_gui_model.py`:
```python
import pytest
from game_agent.gui_model import UsageTracker, format_event, preview_path

def test_format_event_known_types():
    s = format_event({"event": "decision", "observation_id": "obs-1",
                      "decision": {"state": "menu", "reason": "start visible",
                                   "action": {"type": "click", "x": 1, "y": 2}}})
    assert "decision" in s and "start visible" in s and "click" in s

def test_format_event_unknown_type_is_safe():
    assert isinstance(format_event({"event": "weird", "x": 1}), str)

def test_preview_path_prefers_grid(tmp_path):
    raw = tmp_path / "shot.png"; raw.write_bytes(b"x")
    (tmp_path / "shot.grid.png").write_bytes(b"y")
    assert preview_path(str(raw)).endswith("shot.grid.png")
    raw2 = tmp_path / "only.png"; raw2.write_bytes(b"x")
    assert preview_path(str(raw2)) == str(raw2)
    assert preview_path(None) is None

def test_usage_tracker_sums_and_costs():
    t = UsageTracker()
    t.add({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
    t.add({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    assert (t.prompt_tokens, t.completion_tokens, t.total_tokens) == (110, 55, 165)
    assert t.cost_usd(1_000_000, 2_000_000) == pytest.approx(0.110 + 0.110)
    assert t.cost_usd(None, None) is None
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv\Scripts\python -m pytest tests/test_gui_model.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `gui_model.py`**

Pure functions/classes, no Tk, no I/O beyond checking file existence in `preview_path`. `format_event` renders each known event type to a compact multi-line string and falls back to a generic `str(event)` for unknown types. `UsageTracker` sums the three token fields (tolerant of missing) and computes cost only when both prices are provided.

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python -m pytest tests/test_gui_model.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/gui_model.py tests/test_gui_model.py
git commit -m "feat: add GUI display model (events, preview path, usage)"
```

---

### Task 4: `gui_runner.py` (background runner + event queue)

**Files:**
- Create: `src/game_agent/gui_runner.py`
- Create: `tests/test_gui_runner.py`

**Interfaces:**
- Consumes: `Session`, `SessionLogger`, `BrowserController`, `VLMClient`, `Settings`.
- Produces: `GuiRunner(queue, *, controller_factory=None, vlm_factory=None, session_kwargs=None)` with `start(url, goal, settings, viewport=None, grid_step=100)`, `pause()`, `resume()`, `stop()`, `join(timeout=None)`, `is_running` property; a `_QueueLogger(SessionLogger)` that also pushes each event dict to the queue.

- [ ] **Step 1: Write the failing test**

`tests/test_gui_runner.py` (fakes reused from `tests/test_session.py`):
```python
import queue
import pytest
from game_agent.config import Settings
from game_agent.gui_runner import GuiRunner
from games_helpers import *  # noqa  (or import the fakes from tests.test_session)
from tests.test_session import FakeController, FakeVLM, dec, make_sleep

def _events(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out

def test_runner_runs_and_emits_events(tmp_path, clock):
    q = queue.Queue()
    frames = [b"A", b"B"]
    decisions = [dec({"type": "click", "x": 1, "y": 1}), dec({"type": "done", "summary": "ok"})]
    runner = GuiRunner(q,
        controller_factory=lambda url, settings: FakeController(tmp_path, frames),
        vlm_factory=lambda settings: FakeVLM(decisions),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    kinds = [e["event"] for e in _events(q)]
    assert "session_end" in kinds
    assert "observation" in kinds and "decision" in kinds and "action" in kinds

def test_stop_closes_browser(tmp_path, clock):
    q = queue.Queue()
    controllers = []
    def cf(url, settings):
        c = FakeController(tmp_path, [b"A", b"B", b"C"]); controllers.append(c); return c
    runner = GuiRunner(q, controller_factory=cf,
        vlm_factory=lambda s: FakeVLM([dec({"type": "click", "x": 1, "y": 1})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g2.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.stop()
    runner.join(timeout=15)
    assert controllers and controllers[0].closed is True
```

(The implementer must make `FakeController` in `tests/test_session.py` record a `closed` flag and expose `close()`; if it lacks one, add it there minimally — it is a test fake.)

- [ ] **Step 2: Run to verify fail**

Run: `.venv\Scripts\python -m pytest tests/test_gui_runner.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `gui_runner.py`**

- `_QueueLogger(SessionLogger)`: subclass; after each `log_*`, push the emitted record dict onto the queue (build the dict the same way the logger does, or serialize/parse — keep it simple by constructing the dict and passing it to `super`). Simplest: override each method to build the `event` dict once, write it via the parent `_write`, and `queue.put(dict)`.
- `GuiRunner`: holds `queue`, factories, `session_kwargs`; `start` spawns a `threading.Thread` running `asyncio.run(self._run(...))`.
  - `_run`: build controller (`controller_factory(url, settings, viewport)`), `await open()`, `await goto(url)`, build vlm, build `Session(controller, vlm, settings, logger=_QueueLogger(path, queue), grid_step=grid_step, **session_kwargs)`; put `{"event":"status","status":"running"}`; `result = await session.run(goal, ...)`; put `{"event":"session_end","state":str(result.state),"cycles":result.cycles,"reason":result.reason}`; on exception put `{"event":"error","message":str(exc)}`; `finally`: if `self._stop_requested` → `await controller.release_all(); await controller.close()` (else leave open).
  - `pause/resume/stop`: set `self._session.pause()/resume()` and `self._stop_requested`/`self._session.stop()`.
- The `controller_factory` default builds the real `BrowserController` with headed `BrowserOptions` and the given viewport.

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python -m pytest tests/test_gui_runner.py -v` then `.venv\Scripts\python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/gui_runner.py tests/test_gui_runner.py tests/test_session.py
git commit -m "feat: add GUI background runner with event queue"
```

---

### Task 5: `gui.py` (Tk view) + entrypoint + docs

**Files:**
- Create: `src/game_agent/gui.py`
- Modify: `pyproject.toml` (register `gui` marker; keep `addopts` excluding it)
- Modify: `README.md`

**Interfaces:**
- Produces: `python -m game_agent.gui` opens the window; `main()` builds the Tk app.

- [ ] **Step 1: Register the `gui` marker**

In `pyproject.toml` `[tool.pytest.ini_options]`: `addopts = "-m \"not browser and not live and not gui\""` and add `"gui: tests that need a display"` to `markers`.

- [ ] **Step 2: Implement `gui.py`**

A Tk app:
- Top row: `URL` entry (default `examples/demo.html` as a `file://` URI), `Goal` entry, `W`/`H` viewport entries (default 1280/720), buttons `Start`/`Pause`/`Resume`/`Stop`.
- Middle: left = preview `Label` (image), right = read-only `ScrolledText` log.
- Bottom: status label (state / cycles / remaining / tokens / cost).
- `Start`: create a `queue.Queue`, a `GuiRunner`, call `runner.start(url, goal, settings, viewport, grid_step=100)`.
- `after(150, self._poll)`: drain the queue → append `format_event(e)` to the log; on `observation` → load `preview_path(...)` via Pillow → `ImageTk.PhotoImage` (scaled, keep aspect) and set the label; on `model` → update `UsageTracker` + status; on `status`/`session_end`/`error` → update status/log.
- `Stop`: `runner.stop()`; window close (`WM_DELETE_WINDOW`) → `runner.stop()` + destroy.
- `if __name__ == "__main__": main()`.
- Never display the API key.

- [ ] **Step 3: Verify it constructs without a live run**

Run (smoke, manual — needs a display): `python -m game_agent.gui`. Confirm the window opens; close it. (Do NOT start a live run in this step.)
Also confirm the default suite still deselects GUI tests: `.venv\Scripts\python -m pytest -q` → PASS, `gui` marker excluded.

- [ ] **Step 4: Update `README.md`**

Add a **Desktop GUI (M5)** section: `python -m game_agent.gui`, what the window shows (URL/goal/viewport, Start/Pause/Resume/Stop, preview, log, status, tokens/cost), the optional `NVIDIA_PRICE_INPUT_PER_1M`/`NVIDIA_PRICE_OUTPUT_PER_1M` env vars, and that it opens its own headed browser. Add `gui.py`/`gui_model.py`/`gui_runner.py` to the module map.

- [ ] **Step 5: Run the full default suite**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/game_agent/gui.py pyproject.toml README.md
git commit -m "feat: add Tkinter desktop GUI and document M5"
```

---

## Self-Review

- **Spec coverage:** core additions ↔ Tasks 1–2 (§7.2); `gui_model` ↔ Task 3 (§7.5); runner/queue/lifecycle ↔ Task 4 (§7.3, §7.4); Tk view/entrypoint/marker/docs ↔ Task 5 (§7.5, §7.6, §7.7); tests policies ↔ each task + Task 5 (§7.8).
- **Type consistency:** `VLMClient.model`/`last_usage`, `SessionLogger.log_model(observation, model, usage)`, `Session.pause()/resume()/state/cycles/remaining_actions`, `format_event`, `preview_path`, `UsageTracker`, `GuiRunner(queue, *, controller_factory, vlm_factory, session_kwargs)` + `start/pause/resume/stop/join` are named identically wherever produced/consumed.
- **Review Focus:** items 1–5 pinned in Tasks 4/5, 4, 2, 1, 3 respectively.
- **Deferred (design §9):** layout/colors/fonts, GUI `max_cycles`/poll interval, no session-config persistence.
