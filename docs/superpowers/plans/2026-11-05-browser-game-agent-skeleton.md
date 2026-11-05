# Browser Game Agent — Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tested core of a visual browser-game agent — action schema, validator, state machine, session state, and JSONL logging — plus async stubs for the Playwright controller and VLM client, with no browser or API calls.

**Architecture:** Pure, side-effect-free core modules (`schema`, `observation`, `validation`, `state`, `logging`, `config`) around one mutable `SessionState` that owns all cross-action counters and an injected clock. `controller` (Playwright) and `vlm` (HTTP) are async interface stubs implemented in later milestones. No loop driver exists yet.

**Tech Stack:** Python 3.14, Pydantic v2, pytest (+ pytest-asyncio for the stub interfaces only), Playwright and python-dotenv declared but not exercised in tests.

**Spec:** `docs/superpowers/specs/2026-11-05-browser-game-agent-design.md`

## Global Constraints

- Python interpreter: the machine's `python` = **3.14.7**; create the environment with `python -m venv .venv` (no `uv`/`pipx` on this machine).
- Runtime deps: `playwright`, `pydantic>=2`, `python-dotenv`. Dev deps: `pytest`, `pytest-asyncio`.
- Tests MUST run offline: no browser launch, no network, no API keys, no `sleep`.
- Default viewport **1280x720** CSS px; all coordinates are CSS pixels.
- Never write secrets, cookies, tokens, or authorization headers to any log.
- `.env` and runtime output dirs (`screenshots/`, `logs/`, `.venv/`) stay out of Git.
- OpenCode on this machine is **v2.2.0** (MCP schema V2) — not touched by this plan.

## Review Focus

Input classes and failure modes the spec implies but no task's happy path covers, most likely first:

1. **Hostile/malformed VLM output** (extra fields, wrong types, `confidence` outside 0–1, missing `type`): must fail at schema parse and never reach the controller. Pinned in Task 3.
2. **Coordinate boundaries**: `x`/`y` exactly `0`, exactly `viewport_width`/`height`, negative, and non-integer — must be accepted at the closed edge and rejected outside. Pinned in Task 6.
3. **`hold` interrupted mid-flight**: an exception during the wait must still trigger a safe `release`, and held-state must not dangle. Guaranteed by the release invariant documented in Task 2 and exercised by the controller stub contract in Task 9.
4. **Action/rate/budget exhaustion exactly at the threshold** (off-by-one): the boundary action must be the first rejected. Pinned in Task 6.
5. **Repeated failing verification**: consecutive failures must reach the cap and force `NEEDS_USER`, never loop forever. Pinned in Task 7.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/game_agent/__init__.py`
- Create: `tests/__init__.py` (empty)
- Create: `README.md` (one line, expanded in Task 10)

**Interfaces:**
- Consumes: nothing.
- Produces: an installable package `game_agent` importable as `import game_agent`; `pytest` runnable at repo root.

- [ ] **Step 1: Initialize git and declare the package**

Run:
```bash
git init
```
Then write `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "game-agent"
version = "0.1.0"
description = "Visual browser game AI agent"
requires-python = ">=3.14"
dependencies = ["playwright", "pydantic>=2", "python-dotenv"]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Write `.gitignore` and `.env.example`**

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
logs/
screenshots/
output/
.pytest_cache/
.superpowers/
```
`.env.example`:
```
# Game agent configuration overrides (copy to .env; never commit .env)
GAME_AGENT_VIEWPORT_WIDTH=1280
GAME_AGENT_VIEWPORT_HEIGHT=720
GAME_AGENT_CONFIDENCE_THRESHOLD=0.55
# VLM provider credentials are added in Milestone 2
```

- [ ] **Step 3: Create the venv and install the package**

Run:
```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```
Expected: install completes and `pydantic` resolves to a 2.x version. If a wheel is missing for cp314, stop and report before changing anything.

- [ ] **Step 4: Create the package init and placeholder README**

`src/game_agent/__init__.py`:
```python
"""Visual browser game AI agent."""
```
`tests/__init__.py`: empty file.
`README.md`: `# Auto-Browser — visual browser game AI agent`

- [ ] **Step 5: Verify import and empty test run**

Run:
```bash
.venv\Scripts\python -c "import game_agent; print(game_agent.__doc__)"
.venv\Scripts\python -m pytest -q
```
Expected: the docstring prints; pytest exits 0 reporting no tests collected.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example src tests README.md docs
git commit -m "chore: scaffold game_agent package"
```

---

### Task 2: Configuration (`config.py`)

**Files:**
- Create: `src/game_agent/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings` (frozen dataclass) and `load_settings(env=None) -> Settings`. Fields: `viewport_width:int`, `viewport_height:int`, `device_scale_factor:float`, `max_actions_per_session:int`, `min_action_interval_ms:int`, `max_wait_ms:int`, `max_hold_ms:int`, `max_wheel_delta:int`, `confidence_threshold:float`, `allowed_keys:frozenset[str]`, `allowed_buttons:frozenset[str]`.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from game_agent.config import Settings, load_settings

def test_defaults():
    s = load_settings(env={})
    assert s.viewport_width == 1280
    assert s.viewport_height == 720
    assert s.confidence_threshold == 0.55
    assert "left" in s.allowed_buttons

def test_env_override():
    s = load_settings(env={"GAME_AGENT_VIEWPORT_WIDTH": "1920",
                           "GAME_AGENT_CONFIDENCE_THRESHOLD": "0.8"})
    assert s.viewport_width == 1920
    assert s.confidence_threshold == 0.8

def test_settings_is_frozen():
    s = Settings()
    try:
        s.viewport_width = 100  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Settings must be immutable")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_agent.config'`.

- [ ] **Step 3: Implement `config.py`**

`Settings` is a `@dataclass(frozen=True)` with the defaults from Global Constraints. `load_settings(env=None)` reads `env or os.environ`, applies `GAME_AGENT_<FIELD_UPPER>` overrides, and coerces int/float. `allowed_keys` defaults to a set of common game keys (`w a s d up down left right space enter escape shift tab`), `allowed_buttons` to `{"left","right","middle"}`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/config.py tests/test_config.py
git commit -m "feat: add Settings and load_settings"
```

---

### Task 3: Action schema (`schema.py`)

**Files:**
- Create: `src/game_agent/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Button` enum (`left`,`right`,`middle`); action classes `ClickAction`, `MoveAction`, `PressAction`, `HoldAction`, `ReleaseAction`, `DragAction`, `WheelAction`, `WaitAction`, `DoneAction`, `UncertainAction`; `Action` (discriminated union on `type`); `Decision` with `state:str`, `reason:str`, `action:Action`, `expected_result:str|None`, `confidence:float`, `observation_id:str|None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_schema.py`:
```python
import math
import pytest
from pydantic import ValidationError
from game_agent.schema import Decision, ClickAction, HoldAction, UncertainAction

def _decision(**over):
    base = {"state": "menu", "reason": "start visible", "confidence": 0.9,
            "action": {"type": "click", "x": 640, "y": 620}}
    base.update(over)
    return base

def test_valid_click_decision():
    d = Decision.model_validate(_decision())
    assert isinstance(d.action, ClickAction)
    assert d.action.button.value == "left"
    assert d.action.click_count == 1

def test_discriminator_picks_hold():
    d = Decision.model_validate(_decision(action={"type": "hold", "key": "space",
                                                  "duration_ms": 500}))
    assert isinstance(d.action, HoldAction)

def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "click", "x": 1, "y": 2,
                                                  "evil": True}))

def test_unknown_type_rejected():
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "exec", "code": "rm -rf /"}))

def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(confidence=1.5))
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(confidence=math.nan))

def test_click_count_bounds():
    with pytest.raises(ValidationError):
        Decision.model_validate(_decision(action={"type": "click", "x": 1, "y": 2,
                                                  "click_count": 4}))

def test_decision_is_frozen():
    d = Decision.model_validate(_decision())
    with pytest.raises(ValidationError):
        d.confidence = 0.1  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_agent.schema'`.

- [ ] **Step 3: Implement `schema.py`**

All action classes inherit a `_Base(BaseModel)` with `model_config = ConfigDict(extra="forbid", frozen=True)` and a `type: Literal[...]` field. Structural bounds only (no viewport — that is the validator's job): `click_count` 1–3, durations `> 0`, `confidence` `ge=0.0, le=1.0` (NaN fails `ge`/`le`), `key`/`reason`/`summary` non-empty. `HoldAction` and `ReleaseAction` require **exactly one** of `key`/`button` enforced by a `model_validator(mode="after")`. `Action = Annotated[Union[...], Field(discriminator="type")]` including all ten types, so unknown `type` fails discrimination. `Decision` uses the same frozen/extra-forbid config.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_schema.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/schema.py tests/test_schema.py
git commit -m "feat: add discriminated action schema and Decision"
```

---

### Task 4: Observation value type (`observation.py`)

**Files:**
- Create: `src/game_agent/observation.py`
- Test: `tests/test_observation.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Observation` frozen dataclass with `observation_id:str`, `screenshot_path:str|None`, `viewport_width:int`, `viewport_height:int`, `device_scale_factor:float`, `timestamp:float`.

- [ ] **Step 1: Write the failing test**

`tests/test_observation.py`:
```python
import pytest
from game_agent.observation import Observation

def test_observation_fields_and_immutability():
    o = Observation(observation_id="obs-1", screenshot_path="screenshots/1.png",
                    viewport_width=1280, viewport_height=720,
                    device_scale_factor=1.0, timestamp=12.5)
    assert o.observation_id == "obs-1"
    assert o.viewport_width == 1280
    with pytest.raises(Exception):
        o.observation_id = "obs-2"  # type: ignore[misc]

def test_screenshot_path_optional():
    o = Observation(observation_id="obs-2", screenshot_path=None,
                    viewport_width=1280, viewport_height=720,
                    device_scale_factor=1.0, timestamp=0.0)
    assert o.screenshot_path is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_observation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_agent.observation'`.

- [ ] **Step 3: Implement `observation.py`**

A single `@dataclass(frozen=True)` with the six fields above, all typed, no methods.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_observation.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/observation.py tests/test_observation.py
git commit -m "feat: add Observation value type"
```

---

### Task 5: Session state (`session_state.py`)

**Files:**
- Create: `src/game_agent/session_state.py`
- Test: `tests/test_session_state.py`
- Modify: `tests/conftest.py` (create) — shared fake clock fixture

**Interfaces:**
- Consumes: `Settings` (Task 2), `Action`/`HoldAction` (Task 3).
- Produces: `Clock` protocol (`now(self) -> float`); `FakeClock` (test helper, defined in `tests/conftest.py`); `StateSnapshot` frozen dataclass with `action_count:int`, `accumulated_hold_ms:int`, `consecutive_failures:int`, `last_action_at:float|None`; `SessionState` with `__init__(self, settings: Settings, clock: Clock | None = None)`, properties `action_count`, `accumulated_hold_ms`, `consecutive_failures`, `last_action_at`, method `snapshot(self) -> StateSnapshot`, `record(self, action: Action, now: float) -> None`, `record_failure(self) -> None`, `record_success(self) -> None`, `remaining_action_budget(self) -> int`.

- [ ] **Step 1: Write the fake clock fixture and failing tests**

`tests/conftest.py`:
```python
import pytest

class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t
    def now(self) -> float:
        return self.t
    def advance(self, seconds: float) -> None:
        self.t += seconds

@pytest.fixture
def clock():
    return FakeClock()
```

`tests/test_session_state.py`:
```python
from game_agent.config import Settings
from game_agent.schema import ClickAction, HoldAction
from game_agent.session_state import SessionState

def test_record_counts_and_budget(clock):
    s = SessionState(Settings(max_actions_per_session=2), clock=clock)
    assert s.remaining_action_budget() == 2
    s.record(ClickAction(type="click", x=1, y=1), clock.now())
    assert s.action_count == 1
    assert s.remaining_action_budget() == 1

def test_hold_accumulates(clock):
    s = SessionState(Settings(), clock=clock)
    s.record(HoldAction(type="hold", key="w", duration_ms=400), clock.now())
    s.record(HoldAction(type="hold", key="w", duration_ms=600), clock.now())
    assert s.accumulated_hold_ms == 1000

def test_failures_reset_on_success(clock):
    s = SessionState(Settings(), clock=clock)
    s.record_failure(); s.record_failure()
    assert s.consecutive_failures == 2
    s.record_success()
    assert s.consecutive_failures == 0

def test_snapshot_is_immutable_copy(clock):
    s = SessionState(Settings(), clock=clock)
    snap = s.snapshot()
    s.record(ClickAction(type="click", x=1, y=1), clock.now())
    assert snap.action_count == 0

def test_last_action_at_tracks_clock(clock):
    s = SessionState(Settings(), clock=clock)
    clock.advance(5.0)
    s.record(ClickAction(type="click", x=1, y=1), clock.now())
    assert s.last_action_at == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_session_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_agent.session_state'`.

- [ ] **Step 3: Implement `session_state.py`**

`Clock` is a `typing.Protocol` with `now() -> float`. `StateSnapshot` is `@dataclass(frozen=True)`. `SessionState` defaults `clock` to a real monotonic clock (`time.monotonic`) when `None`. `record` increments `action_count`, adds `duration_ms` to `accumulated_hold_ms` only when the action is a `HoldAction`, and sets `last_action_at = now`. `snapshot()` returns a `StateSnapshot` built from current fields.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_session_state.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/session_state.py tests/conftest.py tests/test_session_state.py
git commit -m "feat: add SessionState with injected clock"
```

---

### Task 6: Action validator (`validation.py`)

**Files:**
- Create: `src/game_agent/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `Settings` (Task 2), `Decision`/`Action`/`HoldAction`/`UncertainAction` (Task 3), `StateSnapshot` (Task 5).
- Produces: `RejectedReason` enum (`OUT_OF_BOUNDS`,`NOT_ALLOWED`,`LIMIT_EXCEEDED`,`RATE_LIMITED`,`BUDGET_EXCEEDED`,`LOW_CONFIDENCE`); `ValidationResult` frozen dataclass with `accepted:bool` and `reason:RejectedReason|None`; `validate(decision: Decision, snapshot: StateSnapshot, now: float, settings: Settings) -> ValidationResult` (pure, no side effects).

- [ ] **Step 1: Write the failing tests**

`tests/test_validation.py`:
```python
import pytest
from game_agent.config import Settings
from game_agent.schema import Decision
from game_agent.session_state import StateSnapshot
from game_agent.validation import RejectedReason, validate

SETTINGS = Settings(viewport_width=1280, viewport_height=720,
                    min_action_interval_ms=200, max_actions_per_session=2,
                    max_hold_ms=1000, max_wheel_delta=100, confidence_threshold=0.5)

def _snap(count=0, hold=0, fails=0, last=None):
    return StateSnapshot(action_count=count, accumulated_hold_ms=hold,
                         consecutive_failures=fails, last_action_at=last)

def _dec(action, conf=0.9):
    return Decision.model_validate({"state":"s","reason":"r","confidence":conf,"action":action})

def test_accepts_valid_click():
    assert validate(_dec({"type":"click","x":10,"y":10}), _snap(), now=10.0, settings=SETTINGS).accepted

def test_accepts_coordinates_on_the_closed_edge():
    r = validate(_dec({"type":"click","x":1280,"y":720}), _snap(), now=10.0, settings=SETTINGS)
    assert r.accepted

def test_rejects_out_of_bounds():
    assert validate(_dec({"type":"click","x":-1,"y":10}), _snap(), now=10.0, settings=SETTINGS).reason == RejectedReason.OUT_OF_BOUNDS
    assert validate(_dec({"type":"click","x":10,"y":721}), _snap(), now=10.0, settings=SETTINGS).reason == RejectedReason.OUT_OF_BOUNDS

def test_rejects_key_not_in_allowlist():
    assert validate(_dec({"type":"press","key":"f13"}), _snap(), now=10.0, settings=SETTINGS).reason == RejectedReason.NOT_ALLOWED

def test_rate_limit_boundary():
    # last action at t=10.0, min interval 200ms. Use binary-exact deltas (0.125s, 0.25s)
    # so the boundary assertion is not corrupted by float rounding.
    assert validate(_dec({"type":"move","x":1,"y":1}), _snap(last=10.0), now=10.125, settings=SETTINGS).reason == RejectedReason.RATE_LIMITED
    assert validate(_dec({"type":"move","x":1,"y":1}), _snap(last=10.0), now=10.25, settings=SETTINGS).accepted

def test_budget_boundary_off_by_one():
    assert validate(_dec({"type":"move","x":1,"y":1}), _snap(count=1), now=10.0, settings=SETTINGS).accepted
    assert validate(_dec({"type":"move","x":1,"y":1}), _snap(count=2), now=10.0, settings=SETTINGS).reason == RejectedReason.BUDGET_EXCEEDED

def test_hold_accumulation_guard():
    assert validate(_dec({"type":"hold","key":"w","duration_ms":500}), _snap(hold=500), now=10.0, settings=SETTINGS).accepted
    assert validate(_dec({"type":"hold","key":"w","duration_ms":600}), _snap(hold=500), now=10.0, settings=SETTINGS).reason == RejectedReason.LIMIT_EXCEEDED

def test_wheel_delta_guard():
    assert validate(_dec({"type":"wheel","delta_y":100}), _snap(), now=10.0, settings=SETTINGS).accepted
    assert validate(_dec({"type":"wheel","delta_y":101}), _snap(), now=10.0, settings=SETTINGS).reason == RejectedReason.LIMIT_EXCEEDED

def test_low_confidence_rejected():
    assert validate(_dec({"type":"click","x":1,"y":1}, conf=0.4), _snap(), now=10.0, settings=SETTINGS).reason == RejectedReason.LOW_CONFIDENCE

def test_done_and_uncertain_bypass_rate_and_budget():
    assert validate(_dec({"type":"done","summary":"won"}), _snap(count=99, last=10.0), now=10.0, settings=SETTINGS).accepted
    assert validate(_dec({"type":"uncertain","reason":"unknown"}), _snap(count=99, last=10.0), now=10.0, settings=SETTINGS).accepted
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_validation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_agent.validation'`.

- [ ] **Step 3: Implement `validation.py`**

Order of checks (first rejection wins): (1) `confidence < settings.confidence_threshold` → `LOW_CONFIDENCE`; (2) `done`/`uncertain` bypass all remaining checks and are accepted; (3) coordinate bounds for `click/move/drag` → `OUT_OF_BOUNDS` when any coordinate is `< 0` or `> viewport_*`; `wheel` instead checks `abs(delta) <= max_wheel_delta` and rejects with `LIMIT_EXCEEDED`; (4) allowlist for keys (`press`/`hold`/`release`) and buttons (`click`/`drag`/`hold`/`release`) → `NOT_ALLOWED`; (5) duration guards (`wait` ≤ `max_wait_ms`, `hold` accumulated + this hold ≤ `max_hold_ms`) → `LIMIT_EXCEEDED`; (6) rate limit (`last_action_at is None` passes, else reject when `now - last_action_at < min_action_interval_ms / 1000`) → `RATE_LIMITED`; (7) budget (`action_count >= max_actions_per_session` → `BUDGET_EXCEEDED`). Return `ValidationResult`. No mutation of `snapshot` or any global.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_validation.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/validation.py tests/test_validation.py
git commit -m "feat: add pure ActionValidator"
```

---

### Task 7: State machine (`state.py`)

**Files:**
- Create: `src/game_agent/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: nothing (uses a plain monotonic default clock internally if needed).
- Produces: `State` enum (`IDLE`,`OBSERVING`,`DECIDING`,`VALIDATING`,`ACTING`,`VERIFYING`,`RECOVERING`,`DONE`,`NEEDS_USER`,`STOPPED`); `InvalidTransition(Exception)`; `StateMachine` with `__init__(self, max_consecutive_failures: int = 3)`, `state -> State`, `can(self, target: State) -> bool`, `transition(self, target: State) -> None`, `note_failure(self) -> None`, `note_success(self) -> None`, `stop(self) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_state.py`:
```python
import pytest
from game_agent.state import InvalidTransition, State, StateMachine

def test_happy_path():
    m = StateMachine()
    assert m.state == State.IDLE
    for s in [State.OBSERVING, State.DECIDING, State.VALIDATING, State.ACTING,
              State.VERIFYING, State.OBSERVING]:
        m.transition(s)
    assert m.state == State.OBSERVING

def test_invalid_transition_raises():
    m = StateMachine()
    with pytest.raises(InvalidTransition):
        m.transition(State.ACTING)

def test_done_is_reachable_from_validating():
    m = StateMachine()
    m.transition(State.OBSERVING); m.transition(State.DECIDING); m.transition(State.VALIDATING)
    m.transition(State.DONE)
    assert m.state == State.DONE

def test_failures_cap_forces_needs_user():
    m = StateMachine(max_consecutive_failures=2)
    m.transition(State.OBSERVING); m.transition(State.DECIDING)
    m.transition(State.VALIDATING); m.transition(State.ACTING); m.transition(State.VERIFYING)
    m.note_failure()  # 1
    m.transition(State.RECOVERING)
    m.transition(State.OBSERVING)  # recovery succeeded, back to observing (spec §11)
    for s in [State.DECIDING, State.VALIDATING, State.ACTING, State.VERIFYING]:
        m.transition(s)
    m.note_failure()  # 2 -> cap
    m.transition(State.NEEDS_USER)
    assert m.state == State.NEEDS_USER
    m.transition(State.OBSERVING)  # human resumed
    assert m.state == State.OBSERVING

def test_recovering_and_needs_user_have_exits():
    m = StateMachine()
    for s in [State.OBSERVING, State.DECIDING, State.VALIDATING, State.ACTING, State.RECOVERING]:
        m.transition(s)
    assert m.can(State.OBSERVING)
    assert m.can(State.NEEDS_USER)
    assert m.can(State.STOPPED)

def test_idle_can_reach_observing_and_stopped():
    m = StateMachine()
    assert m.can(State.OBSERVING)
    assert m.can(State.STOPPED)

def test_stop_is_terminal():
    m = StateMachine()
    m.stop()
    assert m.state == State.STOPPED
    with pytest.raises(InvalidTransition):
        m.transition(State.OBSERVING)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_agent.state'`.

- [ ] **Step 3: Implement `state.py`**

Encode the transition table from the spec exactly:
`IDLE→{OBSERVING,STOPPED}`, `OBSERVING→{DECIDING,NEEDS_USER,STOPPED}`,
`DECIDING→{VALIDATING,NEEDS_USER,STOPPED}`, `VALIDATING→{ACTING,OBSERVING,DONE,NEEDS_USER,STOPPED}`,
`ACTING→{VERIFYING,RECOVERING,STOPPED}`, `VERIFYING→{OBSERVING,DONE,RECOVERING,NEEDS_USER,STOPPED}`,
`RECOVERING→{OBSERVING,NEEDS_USER,STOPPED}`, `DONE→{STOPPED}`, `NEEDS_USER→{OBSERVING,STOPPED}`, `STOPPED→{}`.
`note_failure()` increments the counter; `note_success()` resets it. `transition` raises `InvalidTransition(target)` when the pair is absent. `stop()` sets `STOPPED` directly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_state.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/state.py tests/test_state.py
git commit -m "feat: add StateMachine with guarded transitions"
```

---

### Task 8: Session logger (`logging.py`)

**Files:**
- Create: `src/game_agent/logging.py`
- Test: `tests/test_logging.py`

**Interfaces:**
- Consumes: `Observation` (Task 4), `Decision`/`Action` (Task 3), `ValidationResult` (Task 6).
- Produces: `SessionLogger(path: str | Path, clock: Clock | None = None)` with `log_observation(self, obs: Observation) -> None`, `log_decision(self, obs: Observation, decision: Decision) -> None`, `log_action(self, obs: Observation, action: Action, result: ValidationResult | None = None) -> None`, `close(self) -> None`. Each record is one JSON object per line containing at least `event`, `timestamp`, and `observation_id`.

- [ ] **Step 1: Write the failing tests**

`tests/test_logging.py`:
```python
import json
from game_agent.logging import SessionLogger
from game_agent.observation import Observation
from game_agent.schema import Decision

OBS = Observation(observation_id="obs-1", screenshot_path="s1.png",
                  viewport_width=1280, viewport_height=720,
                  device_scale_factor=1.0, timestamp=1.0)

def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def test_logs_observation_and_decision_linked_by_observation_id(tmp_path, clock):
    log = SessionLogger(tmp_path / "session.jsonl", clock=clock)
    log.log_observation(OBS)
    d = Decision.model_validate({"state":"s","reason":"r","confidence":0.9,
                                 "action":{"type":"click","x":1,"y":1}})
    log.log_decision(OBS, d)
    log.close()
    rows = _read(tmp_path / "session.jsonl")
    assert [r["event"] for r in rows] == ["observation", "decision"]
    assert all(r["observation_id"] == "obs-1" for r in rows)

def test_log_action_records_validation_result(tmp_path, clock):
    from game_agent.validation import ValidationResult
    log = SessionLogger(tmp_path / "s.jsonl", clock=clock)
    log.log_action(OBS, Decision.model_validate(
        {"state":"s","reason":"r","confidence":0.9,"action":{"type":"click","x":1,"y":1}}).action,
        result=ValidationResult(accepted=False, reason=None))
    log.close()
    row = _read(tmp_path / "s.jsonl")[0]
    assert row["event"] == "action"
    assert row["accepted"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_logging.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_agent.logging'`.

- [ ] **Step 3: Implement `logging.py`**

Open the file for writing on construction (create parent dir if needed). Each `log_*` serializes with `json.dumps(..., ensure_ascii=False)` and a trailing newline. `log_decision` serializes `decision.model_dump(mode="json")`. `log_action` serializes `action.model_dump(mode="json")` plus `accepted`/`reason` when a result is given. Timestamp comes from the injected clock's `now()`. Never write anything not derived from the `Observation`/`Decision`/`Action`/`ValidationResult` passed in.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_logging.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/logging.py tests/test_logging.py
git commit -m "feat: add JSONL SessionLogger"
```

---

### Task 9: Controller and VLM stubs

**Files:**
- Create: `src/game_agent/controller.py`
- Create: `src/game_agent/vlm.py`
- Test: `tests/test_stubs.py`

**Interfaces:**
- Consumes: `Action` (Task 3), `Observation` (Task 4), `Decision` (Task 3).
- Produces: `BrowserController` with async `open(self) -> None`, async `screenshot(self) -> Observation`, async `execute(self, action: Action) -> None`, async `release_all(self) -> None`, async `close(self) -> None`; `VLMClient` with async `decide(self, observation: Observation, goal: str, remaining_actions: int) -> Decision`.

- [ ] **Step 1: Write the failing test**

`tests/test_stubs.py`:
```python
import inspect
import pytest
from game_agent.controller import BrowserController
from game_agent.vlm import VLMClient
from game_agent.observation import Observation
from game_agent.schema import ClickAction

def test_controller_methods_are_async_and_raise():
    c = BrowserController()
    assert inspect.iscoroutinefunction(c.open)
    assert inspect.iscoroutinefunction(c.execute)
    assert inspect.iscoroutinefunction(c.release_all)
    assert inspect.iscoroutinefunction(c.screenshot)

async def test_controller_execute_not_implemented():
    with pytest.raises(NotImplementedError):
        await BrowserController().execute(ClickAction(type="click", x=1, y=1))

async def test_vlm_decide_not_implemented():
    obs = Observation(observation_id="o", screenshot_path=None, viewport_width=1,
                      viewport_height=1, device_scale_factor=1.0, timestamp=0.0)
    with pytest.raises(NotImplementedError):
        await VLMClient().decide(obs, goal="win", remaining_actions=10)

def test_release_all_documented_contract():
    assert "release" in (BrowserController.release_all.__doc__ or "").lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_stubs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_agent.controller'`.

- [ ] **Step 3: Implement the stubs**

Both classes are `async` methods that `raise NotImplementedError("<milestone>")`. `BrowserController.release_all` carries a docstring stating it must release every held key/button and is the safe-recovery primitive the loop calls from `RECOVERING`; `BrowserController` must track held keys/buttons internally when implemented (Milestone 1). `VLMClient.decide` docstring states it must return a schema-valid `Decision` and never arbitrary code (Milestone 2).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_stubs.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/controller.py src/game_agent/vlm.py tests/test_stubs.py
git commit -m "feat: add async controller and VLM stubs"
```

---

### Task 10: README, roadmap, and full suite

**Files:**
- Modify: `README.md`
- Create: `docs/roadmap.md`

**Interfaces:**
- Consumes: everything above.
- Produces: project documentation and a green full test run.

- [ ] **Step 1: Expand `README.md`**

Cover: purpose, current status (skeleton only — no browser/VLM yet), install command (`.venv\Scripts\python -m pip install -e ".[dev]"`), how to run tests (`.venv\Scripts\python -m pytest`), the `src/game_agent` module map, and a link to the design doc and roadmap.

- [ ] **Step 2: Write `docs/roadmap.md`**

Restate the milestone sequence from the spec: M0 (done — audit), skeleton (this plan), M1 browser PoC, M2 VLM PoC, M3 closed-loop `Session`, M4 game integration, M5 optional GUI. One paragraph each with its acceptance criteria copied from the source plan.

- [ ] **Step 3: Run the full suite**

Run: `.venv\Scripts\python -m pytest -v`
Expected: PASS — every test from Tasks 2–9; no network access, no `sleep`.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/roadmap.md
git commit -m "docs: add README and milestone roadmap"
```

---

## Self-Review

- **Spec coverage:** schema ↔ Task 3; action types/`hold`/`release` ↔ Tasks 2 (invariant), 3, 9; validator rules ↔ Task 6; state machine + `recovering` ↔ Task 7; logging ↔ Task 8; config/secrets ↔ Tasks 1–2; async stubs ↔ Task 9; docs ↔ Task 10; offline deterministic tests ↔ Tasks 5–8 (fake clock). No spec section lacks a task.
- **Type consistency:** `StateSnapshot`, `ValidationResult`, `RejectedReason`, `Observation`, `Decision`, `Action`, `Clock`, `Settings` names and field types are identical across the tasks that define and consume them. `session_state` is the only mutable owner; `validation.validate` is pure.
- **Review Focus:** items 1–5 are pinned in Tasks 3, 6, 2/9, 6, and 7 respectively.
- **Open (deferred):** exact allowlist contents (M1), numeric thresholds (M3), `pydantic-settings` vs hand-rolled env (kept hand-rolled).
