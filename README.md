# Auto-Browser — visual browser game AI agent

An agent that plays browser games through the same loop a human uses:
**observe → decide → act → verify**. It watches the game via screenshots, decides with a
vision-language model (VLM), executes mouse/keyboard actions inside a safety-validated envelope,
and logs every step as JSONL for replay and debugging.

## Current status

**Skeleton only.** The safety-critical core is implemented and unit-tested — action schema, action
validator, state machine, session state, JSONL logging, and configuration. The browser controller
(`controller.py`) and VLM client (`vlm.py`) are **async interface stubs** that raise
`NotImplementedError`: there is no browser automation, no VLM API call, and no session loop driver
yet. Those arrive in M1–M3 (see [docs/roadmap.md](docs/roadmap.md)).

## Install

Windows / PowerShell, Python ≥ 3.14:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Run tests

The suite is offline and deterministic: no browser, no network, no API keys, no `sleep` —
cross-action timing is driven by an injected fake clock.

```powershell
.venv\Scripts\python -m pytest
```

## Module map (`src/game_agent`)

| Module | Role |
|---|---|
| `config.py` | Frozen `Settings` + `load_settings`: viewport, action budget, rate limit, hold/wait/wheel limits, confidence threshold, key/button allowlists; `GAME_AGENT_*` env overrides (see `.env.example`) |
| `schema.py` | `Decision` + `Action` — Pydantic v2 discriminated union over ten action types (`click`, `move`, `press`, `hold`, `release`, `drag`, `wheel`, `wait`, `done`, `uncertain`); extra fields and out-of-range values fail at parse |
| `observation.py` | `Observation` value type: `observation_id`, `screenshot_path`, viewport, `device_scale_factor`, `timestamp` |
| `session_state.py` | `SessionState` — the single mutable owner of cross-action counters (action budget, accumulated hold time, consecutive failures, last action time) behind an injected `Clock`; exposes an immutable `StateSnapshot` |
| `validation.py` | `validate(decision, snapshot, now, settings)` — **pure** gate: coordinate bounds, key/button allowlists, rate, budget, hold/wheel limits, confidence → `accepted | rejected(reason)` |
| `state.py` | `StateMachine` — pure transition guard over `idle → observing → deciding → validating → acting → verifying`, plus `recovering`, `done`, `needs_user`, `stopped`; every state has an exit |
| `logging.py` | `SessionLogger` — one JSON object per line, records linked by `observation_id`; never writes secrets |
| `controller.py` | `BrowserController` — async stub for the Playwright executor (M1); will own held key/button tracking and the safe `release_all` recovery primitive |
| `vlm.py` | `VLMClient` — async stub for the vision model client (M2); must return a schema-valid `Decision`, never arbitrary code |

## Documentation

- Design spec: [`docs/superpowers/specs/2026-11-05-browser-game-agent-design.md`](docs/superpowers/specs/2026-11-05-browser-game-agent-design.md)
- Roadmap (M0 → M5): [`docs/roadmap.md`](docs/roadmap.md)
