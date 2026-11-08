# Auto-Browser — visual browser game AI agent

An agent that plays browser games through the same loop a human uses:
**observe → decide → act → verify**. It watches the game via screenshots, decides with a
vision-language model (VLM), executes mouse/keyboard actions inside a safety-validated envelope,
and logs every step as JSONL for replay and debugging.

## Current status

**Milestones 1, 2 and 3 — the browser controller, the VLM client and the closed-loop session — are
implemented.** The safety-critical core (action schema, action validator, state machine, session state,
JSONL logging, configuration) is joined by a real Playwright async controller: it opens a persistent
Chromium context, navigates, screenshots, and executes schema-validated actions inside the safety
envelope, tracking held keys/buttons so `release_all` can unwind them — best-effort; a failed release
stays in the ledger for retry. The real vision-model client ([Vision model (M2)](#vision-model-m2))
turns a screenshot plus a goal into a schema-valid `Decision`. The closed-loop driver
([Closed-loop session (M3)](#closed-loop-session-m3)) now wires observe → decide → validate → act →
verify together, showing the model a coordinate grid, checking each action against a screenshot diff,
and bounding every run with limits that always terminate and always release held input. Still pending:
M4 — pointing the working loop at one specific browser game (see
[docs/roadmap.md](docs/roadmap.md)).

## Install

Windows / PowerShell, Python ≥ 3.14:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Run tests

The default suite is offline and deterministic: no network, no API keys, no `sleep` — cross-action timing
is driven by an injected fake clock. Real-browser and real-model tests exist but are opt-in (see
[Browser PoC](#browser-poc), [Vision model (M2)](#vision-model-m2) and
[Closed-loop session (M3)](#closed-loop-session-m3)).

```powershell
.venv\Scripts\python -m pytest
```

## Browser PoC

The controller drives a real browser, so its tests are opt-in behind the `browser` marker.

Prerequisite — install the Chromium build that Playwright drives (once per venv):

```powershell
.venv\Scripts\python -m playwright install chromium
```

Run the browser tests (they exercise the controller, mostly against [examples/demo.html](examples/demo.html)):

```powershell
.venv\Scripts\python -m pytest -m browser
```

- **The default suite stays offline.** `pyproject.toml` sets `addopts = -m "not browser and not live"`,
  so a plain `pytest` filters the browser tests out unless you ask for them with `-m browser`.
- **Missing Chromium skips, not fails.** If the browser binary is not installed, the browser tests
  skip with a clear message pointing at the install command above — a fresh checkout is never red.

## Vision model (M2)

`VLMClient.decide` sends the screenshot plus the goal to an OpenAI-compatible `chat/completions`
endpoint and returns a schema-valid `Decision` — never arbitrary code. Model output that cannot be
parsed is repaired once and then reported as a terminal `uncertain` decision.

Configuration comes from `NVIDIA_*` variables, read by `load_vlm_config` (`src/game_agent/vlm.py`):

| Variable | Required | Meaning |
|---|---|---|
| `NVIDIA_API_KEY` | yes | Bearer key for the endpoint; never logged or written to disk |
| `NVIDIA_MODEL` | yes | Model name sent as the request's `model` field |
| `NVIDIA_BASE_URL` | no | OpenAI-compatible base URL; defaults to `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_TIMEOUT_S` / `NVIDIA_MAX_TOKENS` / `NVIDIA_TEMPERATURE` | no | Per-call limits, defaulting to 60 s / 512 tokens / 0.0 |

- **`.env` is read.** `load_vlm_config` loads `.env` (see `.env.example`) via `python-dotenv` and
  merges it *beneath* the process environment (shell variables win); `os.environ` is never mutated.
  Copy `.env.example` to `.env` and fill in the key **and** the model — and never commit `.env`.
- **The live test is opt-in.** [tests/test_vlm_live.py](tests/test_vlm_live.py) drives the real
  browser to [examples/demo.html](examples/demo.html), screenshots it, and calls the real API
  (install Chromium first — see [Browser PoC](#browser-poc)):

  ```powershell
  .venv\Scripts\python -m pytest -m "live and browser" -v
  ```

  It skips unless both `NVIDIA_API_KEY` and `NVIDIA_MODEL` resolve — the guard reads them through
  `load_vlm_config`, exactly as the client does, so they may come from the process environment
  **or** `.env`. As with the browser tests, a missing Chromium binary skips the test, never fails.
- **The default suite stays offline.** The `live` marker is filtered out by the same `addopts` as
  the `browser` marker, so a plain `pytest` never touches the network or your API key.

## Closed-loop session (M3)

`Session` (`src/game_agent/session.py`) is the driver that ties the pieces together. It takes an
**already-open** controller and a vision-model client and runs one goal as a loop of
**observe → decide → validate → act → verify**:

1. **Observe** — capture the viewport and draw a coordinate grid onto a *copy* of the screenshot; the
   original capture stays untouched.
2. **Decide** — ask the model for a schema-valid `Decision`
   ([Vision model (M2)](#vision-model-m2)).
3. **Validate** — run the pure validator against the live `SessionState` snapshot; a rejected
   decision ends the run in `NEEDS_USER`.
4. **Act** — execute the action through the controller's own policy gate.
5. **Verify** — capture the screen again and compare it with the screen from step 1.

`run(goal, max_cycles=...)` returns a `SessionResult` (`state`, `cycles`, `reason`, `last_decision`),
and `controller.release_all()` runs on **every** exit — including an exception — so no key or mouse
button is ever left held down.

### Coordinate grid and screenshot-diff verification

- **The grid gives the model coordinates.** `src/game_agent/overlay.py` (`add_coordinate_grid`) uses
  Pillow to draw a labelled grid onto a copy of the screenshot: x-axis labels along the bottom edge,
  y-axis labels along the left edge, and a legend in the top-right corner. Spacing and labels are in
  **CSS pixels**, so the coordinates the model reads are the coordinates the controller clicks. Set
  `grid_step` to change the spacing (100 CSS px by default) or to `None` to turn the grid off.
- **The diff decides whether an action did anything.** The before/after PNGs are compared
  byte-for-byte. An unchanged screen after a screen-affecting action (`click`, `drag`, `wheel`,
  `press`, `hold`) counts as a **failure**; `wait`, `move` and `release` cannot change the pixels and
  are exempt. A changed screen is a success and clears the failure streak. The outcome of every
  comparison is recorded as a `verify` event in the JSONL log.

### Safety limits — every run terminates

| Limit | Where | Effect when reached |
|---|---|---|
| `max_cycles` | `Session.run(goal, max_cycles=...)` | Upper bound on cycles; the run ends in `STOPPED` |
| `max_consecutive_failures` | `Session(...)` (default 3) | Consecutive failures — a failed action, an unchanged screen, a broken capture — end the run in `NEEDS_USER` |
| `repeat_limit` | `Session(...)` (default 3) | The same action ending on the same screen that many times in a row ends the run in `NEEDS_USER` |
| Action budget | `Settings.max_actions_per_session` | The run ends in `STOPPED` once the budget is used up |
| `stop()` | `Session.stop()` | Stops the session at the next cycle boundary (`STOPPED`) |

A model error or an `uncertain` decision also ends the run in `NEEDS_USER` — the loop never spins
forever and never leaves input stuck down.

### The `pillow` dependency

The grid overlay is the only part of the agent that draws images, so M3 added exactly one dependency:
`pillow`. It is an ordinary project dependency, so the normal install already pulls it in:

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
```

### Running the live end-to-end test

[tests/test_session_live.py](tests/test_session_live.py) drives the **whole** loop against a real
headless Chromium and the real model API: it opens [examples/demo.html](examples/demo.html), lets the
session observe, decide, validate, act and verify, and then checks that the session ended in a
terminal state (`DONE` or `NEEDS_USER`) after at least one cycle. It is opt-in like the other live
tests, and it **skips** — never fails — when the credentials or the Chromium build are missing, so a
fresh checkout is never red:

```powershell
.venv\Scripts\python -m pytest -m "live and browser" tests/test_session_live.py -v
```

It needs `NVIDIA_API_KEY` and `NVIDIA_MODEL` (see [Vision model (M2)](#vision-model-m2)) and the
Playwright Chromium build (see [Browser PoC](#browser-poc)).

- **A note on the demo page.** [examples/demo.html](examples/demo.html) keeps its `Start` button on
  screen after a click — only the `clicks=N` counter changes. A model that keeps re-clicking it never
  reaches the "press the up arrow" half of the goal, so the run ends at the cycle limit (`STOPPED`)
  rather than in `DONE`/`NEEDS_USER`. The loop still terminates and still releases its input; the
  assertion just does not accept that third terminal state.

## Configuration

Defaults live in `Settings` (`src/game_agent/config.py`). Each numeric field can be overridden by
setting a `GAME_AGENT_<FIELD_UPPER>` variable in the **process environment** before launching the
agent — for example, in PowerShell:

```powershell
$env:GAME_AGENT_VIEWPORT_WIDTH = "1920"
```

- **Numeric fields only.** Overrides apply to the int/float settings (viewport, scale factor,
  action budget, rate/hold/wait/wheel limits, confidence threshold — which must stay within
  `[0.0, 1.0]`). The key/button allowlists are code-defined for now and ignore environment
  variables; per-game allowlist configuration remains future work.
- **`.env` is not auto-loaded yet.** `load_settings` reads `os.environ` only — there is no dotenv
  loading. [`.env.example`](.env.example) documents the variable names for that planned
  auto-load (`python-dotenv` is already a declared dependency for it), but copying it to `.env`
  today has no effect on `Settings`; set the variables in your shell instead. (The VLM
  configuration *does* read `.env` — see [Vision model (M2)](#vision-model-m2).)

## Module map (`src/game_agent`)

| Module | Role |
|---|---|
| `config.py` | Frozen `Settings` + `load_settings`: viewport, action budget, rate limit, hold/wait/wheel limits, confidence threshold, key/button allowlists; `GAME_AGENT_*` env overrides for the numeric fields only (see [Configuration](#configuration)) |
| `schema.py` | `Decision` + `Action` — Pydantic v2 discriminated union over ten action types (`click`, `move`, `press`, `hold`, `release`, `drag`, `wheel`, `wait`, `done`, `uncertain`); extra fields and out-of-range values fail at parse |
| `observation.py` | `Observation` value type: `observation_id`, `screenshot_path`, viewport, `device_scale_factor`, `timestamp` |
| `session_state.py` | `SessionState` — the single mutable owner of cross-action counters (action budget, accumulated hold time, consecutive failures, last action time) behind an injected `Clock`; exposes an immutable `StateSnapshot` |
| `validation.py` | `validate(decision, snapshot, now, settings)` — **pure** gate: coordinate bounds, key/button allowlists, rate, budget, hold/wheel limits, confidence → `accepted | rejected(reason)` |
| `state.py` | `StateMachine` — pure transition guard over `IDLE → OBSERVING → DECIDING → VALIDATING → ACTING → VERIFYING`, plus `RECOVERING`, `DONE`, `NEEDS_USER`, `STOPPED`; every state has an exit |
| `logging.py` | `SessionLogger` — one JSON object per line, records linked by `observation_id` (`observation`, `decision`, `action`, `verify`); never writes secrets |
| `keys.py` | `to_playwright_key` — pure translation from agent key names (`up`, `space`, …) to Playwright key names (`ArrowUp`, `Space`, …); single characters and unknown names pass through unchanged, so a bad name surfaces as a Playwright error |
| `controller.py` | `BrowserController` — the real async Playwright executor (M1): owns the browser lifecycle (`open`/`close`, one persistent Chromium context sized from `Settings`), `goto`, `screenshot` → `Observation`, and `execute`, which gates every action through `assert_executable` before dispatch. A held-input ledger (`held_keys`/`held_buttons`) backs the safe `release_all` recovery primitive; during `execute`, Playwright failures are wrapped as `ActionExecutionError` (policy violations raise `ActionNotAllowed`), so one `except ControllerError` covers the whole `execute` contract |
| `vlm.py` | `VLMClient` + `load_vlm_config` — the real vision-model client (M2): turns a screenshot data URL plus goal into chat messages, POSTs them to an OpenAI-compatible `chat/completions` endpoint, and parses the reply into a schema-valid `Decision` (never arbitrary code); unparseable output is repaired once, then reported as a terminal `uncertain` decision; `NVIDIA_*` env/`.env` configuration (see [Vision model (M2)](#vision-model-m2)) |
| `overlay.py` | `add_coordinate_grid` — the coordinate-grid overlay (M3): draws a labelled grid onto a **copy** of a screenshot with Pillow; spacing and labels are in CSS pixels (x labels along the bottom edge, y labels along the left edge, legend in the top-right corner), and the source image is never modified |
| `session.py` | `Session` + `SessionResult` — the closed-loop driver (M3): runs observe → decide → validate → act → verify over an already-open controller, shows the model the grid-overlaid screenshot, verifies each action with a screenshot diff, and is bounded by `max_cycles`, the action budget, the consecutive-failure cap, the repeat limit and `stop()`; `release_all()` runs on every exit (see [Closed-loop session (M3)](#closed-loop-session-m3)) |

## Documentation

- Design spec: [`docs/superpowers/specs/2026-11-05-browser-game-agent-design.md`](docs/superpowers/specs/2026-11-05-browser-game-agent-design.md)
- Roadmap (M0 → M5): [`docs/roadmap.md`](docs/roadmap.md)
