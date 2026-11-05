# Roadmap

Milestone sequence for the visual browser game AI agent. The binding research brief
(`browser_game_ai_agent_plan.md`) lives outside this repository; the acceptance criteria below are
restated from it via the design spec
([`docs/superpowers/specs/2026-11-05-browser-game-agent-design.md`](superpowers/specs/2026-11-05-browser-game-agent-design.md)).

## M0 — Environment audit (done)

Before any code, the machine was audited so the plan rests on verified facts: OpenCode v2.2.0 uses
MCP schema V2 (server config under `mcp.servers.<name>` with a `disabled` flag, and the `mcp add`
CLI demonstrably mishandles `--`, so MCP servers must be configured by editing files), Python is
3.14.7 (dependencies need cp314 wheels), no `uv`/`pipx` exist (plain `python -m venv` + `pip`), the
provider config has no OpenRouter (VLM traffic goes through an internal proxy, so the VLM client
must stay provider-abstract), and the Playwright MCP server ships the flags the agent needs
(`--caps vision`, `--isolated`, `--viewport-size`, `--image-responses`, `--snapshot-boxes`) for a
dedicated browser with exact coordinates. **Acceptance:** constraints verified and recorded in the
design spec (§5) before implementation began — done.

## Skeleton — core invariants (done)

This plan (`docs/superpowers/plans/2026-11-05-browser-game-agent-skeleton.md`) built the tested
core before touching Playwright or any VLM API, because the action schema, validator, state
machine, session state, and logs determine how safe and debuggable the whole system is — and they
are all testable without a browser, an API, or money. The controller and VLM client exist as async
interface stubs so later milestones slot in without re-architecture. **Acceptance:** action schema
+ validator + state machine + session state + JSONL logger implemented for real, offline
deterministic unit tests green (fake clock — no browser, network, API keys, or `sleep`), no empty
speculative modules; done.

## M1 — Browser PoC

Replace the `BrowserController` stub with a real Playwright-backed executor that opens a
**dedicated** browser for the agent (isolated profile, fixed viewport), captures screenshots into
`Observation` objects, and executes allowlisted mouse/keyboard actions at real CSS-pixel
coordinates, tracking held inputs and unwinding them via `release_all`. **Acceptance (source
plan):** the agent opens a dedicated browser and can both screenshot the page and act on it —
schema-valid actions land in the game page; the exact key/button allowlist contents deferred from
the skeleton are pinned here once a real page is in front of us.

## M2 — VLM PoC

Select the vision model(s) and prove they can drive the schema. The machine reaches providers
through an internal proxy (no direct OpenRouter), so the client stays provider-abstract and the
VLM SDK is added only here, once the real provider is known. **Acceptance (source plan):** vision
models are selected and structured output is validated — `VLMClient.decide` returns `Decision`s
that parse against the action schema (malformed payloads, extra fields, and out-of-range values
are rejected at parse, never reaching the controller), with confidence and `observation_id`
populated.

## M3 — Closed-loop `Session`

Add the orchestrator that ties observation → decision → validation → action → verification into a
loop owned by a `Session`, using the existing pure validator, mutable `SessionState`, and state
machine as-is. **Acceptance (source plan):** the closed loop runs with the validator, budget, and
emergency stop enforced every cycle — each action is checked against the live `SessionState`
snapshot, the action budget and rate limit gate the loop, consecutive verification failures hit
the retry cap and force `needs_user` (never an infinite loop), and a stop command halts from any
state; the numeric thresholds (budget/rate/confidence) left at sane defaults in the skeleton are
tuned here.

## M4 — Game integration

Point the working loop at one specific browser game. Per-game configuration (goal prompts, key
allowlist extensions, viewport, coordinate conventions) is layered through `Settings` and the MCP
Playwright stack proven in M1; no game logic, coordinates, or tactics are hard-coded into the core
modules. **Acceptance (source plan):** a specific game is integrated — the agent completes
goal-driven sessions in that game end-to-end, with JSONL logs sufficient to replay every decision.

## M5 — Optional GUI

An optional desktop UI on top of a working agent, built only if it earns its keep. **Acceptance
(source plan):** a desktop UI (optional) to observe running sessions and steer them — start/stop,
approve `needs_user` pauses, and view live logs and screenshots. Skipping M5 blocks no core
capability.
