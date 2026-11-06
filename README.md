# Auto-Browser — visual browser game AI agent

An agent that plays browser games through the same loop a human uses:
**observe → decide → act → verify**. It watches the game via screenshots, decides with a
vision-language model (VLM), executes mouse/keyboard actions inside a safety-validated envelope,
and logs every step as JSONL for replay and debugging.

## Current status

**Milestone 1 — the browser controller — is implemented.** The safety-critical core (action schema,
action validator, state machine, session state, JSONL logging, configuration) is joined by a real
Playwright async controller: it opens a persistent Chromium context, navigates, screenshots, and
executes schema-validated actions inside the safety envelope, tracking held keys/buttons so
`release_all` can always unwind them. Still pending: the VLM client (M2) — `vlm.py` remains an
async stub that raises `NotImplementedError` — and the session loop driver (M3) that wires
observe → decide → act → verify together (see [docs/roadmap.md](docs/roadmap.md)).

## Install

Windows / PowerShell, Python ≥ 3.14:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Run tests

The suite is offline and deterministic: no network, no API keys, no `sleep` — cross-action timing
is driven by an injected fake clock. Real-browser tests exist but are opt-in (see
[Browser PoC](#browser-poc)).

```powershell
.venv\Scripts\python -m pytest
```

## Browser PoC

The controller drives a real browser, so its tests are opt-in behind the `browser` marker.

Prerequisite — install the Chromium build that Playwright drives (once per venv):

```powershell
.venv\Scripts\python -m playwright install chromium
```

Run the browser tests (they exercise the controller against [examples/demo.html](examples/demo.html)):

```powershell
.venv\Scripts\python -m pytest -m browser
```

- **The default suite stays offline.** `pyproject.toml` sets `addopts = -m "not browser"`, so a
  plain `pytest` filters the browser tests out unless you ask for them with `-m browser`.
- **Missing Chromium skips, not fails.** If the browser binary is not installed, the browser tests
  skip with a clear message pointing at the install command above — a fresh checkout is never red.

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
  today has no effect; set the variables in your shell instead.

## Module map (`src/game_agent`)

| Module | Role |
|---|---|
| `config.py` | Frozen `Settings` + `load_settings`: viewport, action budget, rate limit, hold/wait/wheel limits, confidence threshold, key/button allowlists; `GAME_AGENT_*` env overrides for the numeric fields only (see [Configuration](#configuration)) |
| `schema.py` | `Decision` + `Action` — Pydantic v2 discriminated union over ten action types (`click`, `move`, `press`, `hold`, `release`, `drag`, `wheel`, `wait`, `done`, `uncertain`); extra fields and out-of-range values fail at parse |
| `observation.py` | `Observation` value type: `observation_id`, `screenshot_path`, viewport, `device_scale_factor`, `timestamp` |
| `session_state.py` | `SessionState` — the single mutable owner of cross-action counters (action budget, accumulated hold time, consecutive failures, last action time) behind an injected `Clock`; exposes an immutable `StateSnapshot` |
| `validation.py` | `validate(decision, snapshot, now, settings)` — **pure** gate: coordinate bounds, key/button allowlists, rate, budget, hold/wheel limits, confidence → `accepted | rejected(reason)` |
| `state.py` | `StateMachine` — pure transition guard over `IDLE → OBSERVING → DECIDING → VALIDATING → ACTING → VERIFYING`, plus `RECOVERING`, `DONE`, `NEEDS_USER`, `STOPPED`; every state has an exit |
| `logging.py` | `SessionLogger` — one JSON object per line, records linked by `observation_id`; never writes secrets |
| `keys.py` | `to_playwright_key` — pure translation from agent key names (`up`, `space`, …) to Playwright key names (`ArrowUp`, `Space`, …); single characters and unknown names pass through unchanged, so a bad name surfaces as a Playwright error |
| `controller.py` | `BrowserController` — the real async Playwright executor (M1): owns the browser lifecycle (`open`/`close`, one persistent Chromium context sized from `Settings`), `goto`, `screenshot` → `Observation`, and `execute`, which gates every action through `assert_executable` before dispatch. A held-input ledger (`held_keys`/`held_buttons`) backs the safe `release_all` recovery primitive; Playwright failures are wrapped as `ActionExecutionError` (policy violations raise `ActionNotAllowed`), so one `except ControllerError` covers the whole contract |
| `vlm.py` | `VLMClient` — async stub for the vision model client (M2); must return a schema-valid `Decision`, never arbitrary code |

## Documentation

- Design spec: [`docs/superpowers/specs/2026-11-05-browser-game-agent-design.md`](docs/superpowers/specs/2026-11-05-browser-game-agent-design.md)
- Roadmap (M0 → M5): [`docs/roadmap.md`](docs/roadmap.md)
