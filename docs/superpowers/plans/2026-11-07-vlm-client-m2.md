# M2 — VLM Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the `VLMClient` stub into a real client that sends a screenshot + goal to NVIDIA's OpenAI-compatible API and returns a schema-validated `Decision`, with a one-shot repair and a safe `uncertain` fallback.

**Architecture:** Config and error type read from the environment (no hard-coded key/model); pure helpers encode the image and build/parse OpenAI-style messages; the async `VLMClient.decide` calls an injectable sync transport (stdlib `urllib` in a thread) so tests stay offline.

**Tech Stack:** Python 3.14, stdlib `urllib.request` + `asyncio.to_thread`, Pydantic v2, python-dotenv, pytest (+ pytest-asyncio). Reuses `Observation`, `Decision`, action schema.

**Spec:** `docs/superpowers/specs/2026-11-07-vlm-client-m2-design.md`

## Global Constraints

- Python 3.14.7; `.venv`. **No new HTTP dependency** — use stdlib `urllib.request` run via `asyncio.to_thread`.
- Credentials/model come from env ONLY: `NVIDIA_API_KEY`, `NVIDIA_MODEL` (both required at call time), optional `NVIDIA_BASE_URL` / `NVIDIA_MAX_TOKENS` / `NVIDIA_TEMPERATURE` / `NVIDIA_TIMEOUT_S`. **Never hard-code** a key or model.
- **Never log** the API key, the raw image, or the full request/response body; error messages must not contain the key.
- Default `pytest` MUST stay offline: register a `live` marker and set `addopts = "-m \"not browser and not live\""`.
- Reuse `Observation`, `Decision`, `UncertainAction`. The client does NOT bound-check viewport coordinates (the validator owns that); parse is strict (extra fields / wrong types rejected) after stripping a code fence only.
- `load_vlm_config(env=...)`: a passed `env` is the ONLY source (deterministic tests); `env=None` reads `.env` + process env (process wins) without mutating `os.environ`.

## Review Focus

Input classes / failure modes most likely to bite, most likely first:

1. **Secret leakage**: the key must never appear in logs, exceptions, or the report; the offline tests must not print it. Pinned in Task 3.
2. **Offline-by-default**: the default suite must never touch the network (no live call). Pinned in Task 1 (marker/addopts) and Task 3 (fake transport).
3. **Env isolation**: `load_vlm_config(env={})` must ignore the process env / `.env`, so an offline test can't accidentally pick up a real key. Pinned in Task 1.
4. **Unusable model output** (markdown fences, extra fields, truncated JSON): must not crash and must not silently pass — strip fence, validate strictly, repair once, then `uncertain`. Pinned in Tasks 2–3.
5. **Missing key/model/screenshot**: a clear `VLMError`, never a raw `KeyError`/`TypeError`/`FileNotFoundError`. Pinned in Task 3.

---

### Task 1: VLM config, error type, and test wiring

**Files:**
- Modify: `src/game_agent/vlm.py`
- Modify: `pyproject.toml` (register `live` marker; update `addopts`)
- Modify: `.env.example` (NVIDIA_* placeholders)
- Create: `tests/test_vlm_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `VLMError(Exception)`; `VLMConfig` frozen dataclass (`base_url="https://integrate.api.nvidia.com/v1"`, `model=None`, `api_key=None`, `timeout_s=60.0`, `max_tokens=512`, `temperature=0.0`); `load_vlm_config(env: Mapping[str,str] | None = None) -> VLMConfig`.

- [ ] **Step 1: Write the failing test**

`tests/test_vlm_config.py`:
```python
import dataclasses
import pytest
from game_agent import vlm
from game_agent.vlm import VLMConfig, load_vlm_config

def test_defaults():
    c = load_vlm_config(env={})
    assert c.base_url == "https://integrate.api.nvidia.com/v1"
    assert c.model is None and c.api_key is None
    assert (c.timeout_s, c.max_tokens, c.temperature) == (60.0, 512, 0.0)

def test_env_override():
    c = load_vlm_config(env={"NVIDIA_API_KEY": "k", "NVIDIA_MODEL": "m",
                             "NVIDIA_BASE_URL": "http://x/v1", "NVIDIA_MAX_TOKENS": "256",
                             "NVIDIA_TEMPERATURE": "0.2", "NVIDIA_TIMEOUT_S": "5"})
    assert (c.api_key, c.model, c.base_url) == ("k", "m", "http://x/v1")
    assert (c.max_tokens, c.temperature, c.timeout_s) == (256, 0.2, 5.0)

def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        load_vlm_config(env={}).model = "x"  # type: ignore[misc]

def test_env_arg_is_the_only_source(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "from-process")
    assert load_vlm_config(env={}).api_key is None

def test_none_reads_process_env(monkeypatch):
    monkeypatch.setattr(vlm, "dotenv_values", lambda *a, **k: {})  # ignore any real .env
    monkeypatch.setenv("NVIDIA_MODEL", "from-process")
    assert load_vlm_config().model == "from-process"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_vlm_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'VLMConfig'`.

- [ ] **Step 3: Implement config + error in `vlm.py`**

Keep the existing `VLMClient` stub below. Add `from dotenv import dotenv_values`, a module-level `VLMError(Exception)`, the `VLMConfig` frozen dataclass with the defaults above, and `load_vlm_config`:
- `env is not None` → read only `env`.
- `env is None` → `base = {**dotenv_values(), **os.environ}` (process wins; no global mutation), then read `base`.
- Map `NVIDIA_API_KEY`→`api_key`, `NVIDIA_MODEL`→`model`, `NVIDIA_BASE_URL`→`base_url`, `NVIDIA_MAX_TOKENS`→`max_tokens` (int), `NVIDIA_TEMPERATURE`→`temperature` (float), `NVIDIA_TIMEOUT_S`→`timeout_s` (float). No default key/model.

- [ ] **Step 4: Register the `live` marker and update `addopts`**

In `pyproject.toml` `[tool.pytest.ini_options]`: change `addopts` to `-m "not browser and not live"` and add `"live: tests that call a real model API (opt-in; run with -m live)"` to `markers`.

- [ ] **Step 5: Add `.env.example` placeholders**

Append:
```
# M2 — vision model (NVIDIA API, OpenAI-compatible). Never commit the real .env.
NVIDIA_API_KEY=
NVIDIA_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b
#NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
#NVIDIA_MAX_TOKENS=512
#NVIDIA_TEMPERATURE=0.0
#NVIDIA_TIMEOUT_S=60
```

- [ ] **Step 6: Run tests + default suite**

Run: `.venv\Scripts\python -m pytest tests/test_vlm_config.py -v` → PASS (5 tests).
Run: `.venv\Scripts\python -m pytest -q` → PASS, offline.

- [ ] **Step 7: Commit**

```bash
git add src/game_agent/vlm.py pyproject.toml .env.example tests/test_vlm_config.py
git commit -m "feat: add VLM config, error type, and live marker"
```

---

### Task 2: Image encoding, message building, and output parsing

**Files:**
- Modify: `src/game_agent/vlm.py`
- Create: `tests/test_vlm_helpers.py`

**Interfaces:**
- Consumes: `Decision`, `UncertainAction` (schema); `VLMConfig`.
- Produces: `encode_image_data_url(path: str | Path) -> str`; `build_messages(image_data_url: str, goal: str, remaining_actions: int, viewport_width: int, viewport_height: int) -> list[dict]`; `parse_decision(content: str, observation_id: str) -> Decision` (raises `pydantic.ValidationError`/`ValueError` on bad output); `fallback_decision(observation_id: str, error: str) -> Decision`.

- [ ] **Step 1: Write the failing test**

`tests/test_vlm_helpers.py`:
```python
import base64
import pytest
from pydantic import ValidationError
from game_agent.schema import Decision
from game_agent.vlm import (build_messages, encode_image_data_url, fallback_decision,
                            parse_decision)

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

def test_encode_image_data_url(tmp_path):
    p = tmp_path / "s.png"
    p.write_bytes(PNG)
    url = encode_image_data_url(p)
    assert url.startswith("data:image/png;base64,")
    assert isinstance(base64.b64decode(url.split(",", 1)[1]), bytes)

def test_build_messages_shape():
    msgs = build_messages("data:image/png;base64,AAAA", "win the level", 7, 1280, 720)
    assert msgs[0]["role"] == "system"
    assert "JSON" in msgs[0]["content"]
    user = msgs[-1]
    assert user["role"] == "user"
    text = [c for c in user["content"] if c["type"] == "text"][0]["text"]
    assert "win the level" in text and "1280" in text and "7" in text
    img = [c for c in user["content"] if c["type"] == "image_url"][0]
    assert img["image_url"]["url"].startswith("data:image/png;base64,")

def test_parse_valid():
    d = parse_decision('{"state":"menu","reason":"go","confidence":0.9,'
                       '"action":{"type":"click","x":10,"y":20}}', "obs-1")
    assert isinstance(d, Decision) and d.observation_id == "obs-1"

def test_parse_strips_code_fence():
    d = parse_decision('```json\n{"state":"m","reason":"r","confidence":0.9,'
                       '"action":{"type":"wait","duration_ms":100}}\n```', "obs-2")
    assert d.action.type == "wait"

def test_parse_rejects_extra_field():
    with pytest.raises((ValidationError, ValueError)):
        parse_decision('{"state":"m","reason":"r","confidence":0.9,'
                       '"action":{"type":"click","x":1,"y":1,"evil":1}}', "obs-3")

def test_parse_rejects_garbage():
    with pytest.raises((ValidationError, ValueError)):
        parse_decision("not json at all", "obs-4")

def test_fallback_fields():
    d = fallback_decision("obs-9", "boom")
    assert d.state == "uncertain" and d.confidence == 1.0
    assert d.action.type == "uncertain" and d.observation_id == "obs-9"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_vlm_helpers.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement the helpers in `vlm.py`**

- `encode_image_data_url`: read bytes, `base64.b64encode(...).decode()`, return `f"data:image/png;base64,{b64}"`.
- `build_messages`: system message describing the exact `Decision` JSON schema + action list + rules (CSS-px coords, `hold` needs `duration_ms`, `state` is a short screen label, `uncertain` when unsure, `done` when the goal is met, output ONLY one JSON object with no prose/fence); user message = a `content` list with a `{"type":"text",...}` part (goal, remaining_actions, viewport) and an `{"type":"image_url","image_url":{"url": image_data_url}}` part.
- `parse_decision`: strip whitespace and an optional leading/trailing ``` fence (and a leading `json` tag), then `Decision.model_validate_json(...)`; set `observation_id` via `model_copy(update={"observation_id": observation_id})`. Let `ValidationError`/`ValueError` propagate.
- `fallback_decision`: build the exact fallback from design §7.4 (`state="uncertain"`, `action=UncertainAction(type="uncertain", reason=...)`, `expected_result=None`, `confidence=1.0`, `observation_id` set).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_vlm_helpers.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/game_agent/vlm.py tests/test_vlm_helpers.py
git commit -m "feat: add VLM image encoding, prompt building, and decision parsing"
```

---

### Task 3: `VLMClient.decide` with injectable transport

**Files:**
- Modify: `src/game_agent/vlm.py`
- Create: `tests/test_vlm_client.py`

**Interfaces:**
- Consumes: everything in Tasks 1–2; `Observation`; `Decision`.
- Produces: `VLMClient(config: VLMConfig | None = None, transport: Callable[[str, dict, dict, float], dict] | None = None)` with `async decide(self, observation: Observation, goal: str, remaining_actions: int) -> Decision`. Default transport = `_post_json(url, headers, payload, timeout_s) -> dict` (stdlib urllib).

- [ ] **Step 1: Write the failing test**

`tests/test_vlm_client.py` (offline; fake transport, no network):
```python
import base64
import pytest
from game_agent.observation import Observation
from game_agent.schema import Decision
from game_agent.vlm import VLMClient, VLMConfig, VLMError

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
VALID = '{"state":"m","reason":"r","confidence":0.9,"action":{"type":"wait","duration_ms":100}}'

def obs(tmp_path, with_shot=True):
    p = None
    if with_shot:
        p = tmp_path / "s.png"; p.write_bytes(PNG)
    return Observation(observation_id="obs-1", screenshot_path=str(p) if p else None,
                       viewport_width=1280, viewport_height=720,
                       device_scale_factor=1.0, timestamp=0.0)

def cfg():
    return VLMConfig(model="test-model", api_key="secret-key")

def reply(content):
    return {"choices": [{"message": {"content": content}}]}

async def test_decide_happy_path(tmp_path):
    calls = {}
    def fake(url, headers, payload, timeout_s):
        calls.update(url=url, headers=headers, payload=payload); return reply(VALID)
    d = await VLMClient(cfg(), transport=fake).decide(obs(tmp_path), "goal", 5)
    assert isinstance(d, Decision) and d.observation_id == "obs-1"
    assert calls["url"].endswith("/chat/completions")
    assert calls["headers"]["Authorization"] == "Bearer secret-key"
    assert calls["payload"]["model"] == "test-model"
    img = [c for c in calls["payload"]["messages"][-1]["content"] if c["type"] == "image_url"][0]
    assert img["image_url"]["url"].startswith("data:image/png;base64,")

async def test_decide_repairs_once(tmp_path):
    seen = []
    def fake(url, headers, payload, timeout_s):
        seen.append(payload["messages"]); return reply("garbage" if len(seen) == 1 else VALID)
    d = await VLMClient(cfg(), transport=fake).decide(obs(tmp_path), "goal", 5)
    assert d.action.type == "wait" and len(seen) == 2
    assert any(m["role"] == "assistant" for m in seen[1])  # bad output echoed

async def test_decide_falls_back_to_uncertain(tmp_path):
    def fake(url, headers, payload, timeout_s):
        return reply("still garbage")
    d = await VLMClient(cfg(), transport=fake).decide(obs(tmp_path), "goal", 5)
    assert d.state == "uncertain" and d.confidence == 1.0 and d.action.type == "uncertain"

async def test_missing_key_raises(tmp_path):
    with pytest.raises(VLMError):
        await VLMClient(VLMConfig(model="m"), transport=lambda *a: reply(VALID)).decide(obs(tmp_path), "g", 1)

async def test_missing_model_raises(tmp_path):
    with pytest.raises(VLMError):
        await VLMClient(VLMConfig(api_key="k"), transport=lambda *a: reply(VALID)).decide(obs(tmp_path), "g", 1)

async def test_missing_screenshot_raises(tmp_path):
    with pytest.raises(VLMError):
        await VLMClient(cfg(), transport=lambda *a: reply(VALID)).decide(obs(tmp_path, with_shot=False), "g", 1)

async def test_protocol_error_raises(tmp_path):
    with pytest.raises(VLMError):
        await VLMClient(cfg(), transport=lambda *a: {"choices": []}).decide(obs(tmp_path), "g", 1)

async def test_transport_error_wrapped(tmp_path):
    def boom(*a): raise RuntimeError("network down")
    with pytest.raises(VLMError):
        await VLMClient(cfg(), transport=boom).decide(obs(tmp_path), "g", 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_vlm_client.py -v`
Expected: FAIL — `decide` still raises `NotImplementedError` / helpers missing.

- [ ] **Step 3: Implement `decide` + default transport**

Replace the `VLMClient` stub. `decide`:
1. Resolve `config = self._config or load_vlm_config()`; require `api_key` and `model` → else `VLMError`.
2. Require `observation.screenshot_path` and the file to exist → else `VLMError` (do NOT leak the path in a way that includes secrets — the path is fine).
3. `image = encode_image_data_url(path)`; `messages = build_messages(image, goal, remaining_actions, obs.viewport_width, obs.viewport_height)`.
4. `payload = {"model": ..., "messages": messages, "max_tokens": ..., "temperature": ...}`; `headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}`; `url = config.base_url.rstrip("/") + "/chat/completions"`.
5. Call `transport = self._transport or _post_json` via `await asyncio.to_thread(transport, url, headers, payload, config.timeout_s)`; wrap any exception that is not already `VLMError` into `VLMError("request to the model endpoint failed")` (message must NOT include the key).
6. `content = _extract_content(resp)` (raise `VLMError` when no usable content string).
7. `try: return parse_decision(content, obs.observation_id)`; on `ValidationError`/`ValueError`, repair once (append the model's bad output as an `assistant` turn and a `user` instruction to return only valid JSON), call again, `try` parse; on second failure return `fallback_decision(obs.observation_id, str(err))`.
- `_post_json(url, headers, payload, timeout_s)`: `urllib.request` POST, JSON body; non-2xx or network error → `VLMError` (message without the key); return `json.loads(body)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_vlm_client.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Verify the default suite is still green and offline**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS — no network, `live`/`browser` deselected.

- [ ] **Step 6: Commit**

```bash
git add src/game_agent/vlm.py tests/test_vlm_client.py
git commit -m "feat: implement VLMClient.decide with repair and uncertain fallback"
```

---

### Task 4: Opt-in live test + README

**Files:**
- Create: `tests/test_vlm_live.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `VLMClient`, `BrowserController` (M1), `examples/demo.html`.
- Produces: a live end-to-end test (browser screenshot → real API → `Decision`) and an M2 README section.

- [ ] **Step 1: Write the live test**

`tests/test_vlm_live.py`:
```python
import os
import pathlib
import pytest
from game_agent.config import Settings
from game_agent.controller import BrowserController, BrowserOptions
from game_agent.schema import Decision
from game_agent.vlm import VLMClient

pytestmark = [pytest.mark.live, pytest.mark.browser]
DEMO = pathlib.Path(__file__).resolve().parents[1] / "examples" / "demo.html"

async def test_live_decide_on_demo_screenshot(tmp_path):
    if not (os.environ.get("NVIDIA_API_KEY") and os.environ.get("NVIDIA_MODEL")):
        pytest.skip("set NVIDIA_API_KEY and NVIDIA_MODEL to run the live test")
    controller = BrowserController(Settings(), options=BrowserOptions(
        headless=True, user_data_dir=str(tmp_path / "p"), screenshot_dir=str(tmp_path / "s")))
    await controller.open()
    try:
        await controller.goto(DEMO.as_uri())
        obs = await controller.screenshot()
        decision = await VLMClient().decide(obs, goal="Find and click the Start button.", remaining_actions=3)
    finally:
        await controller.close()
    assert isinstance(decision, Decision)
    assert decision.observation_id == obs.observation_id
```

- [ ] **Step 2: Run the default suite (live test must be deselected)**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS — the live test is not collected.

- [ ] **Step 3: Run the live test once (real API, approved)**

Run: `$env:NVIDIA_API_KEY="..."; $env:NVIDIA_MODEL="..."; .venv\Scripts\python -m pytest -m "live and browser" tests/test_vlm_live.py -v`
Expected: PASS if key+model valid; otherwise `SKIP`. Record the outcome in the report. If it fails because the model returns unparseable JSON, that is a real signal — report it (the client should still return a `Decision`, possibly `uncertain`).

- [ ] **Step 4: Update `README.md`**

Update **Current status** (VLM client M2 implemented; the loop M3 pending). Add a **Vision model (M2)** section: the `NVIDIA_API_KEY` / `NVIDIA_MODEL` env vars (and `.env` support via python-dotenv), the default base URL, how to run the live test (`pytest -m "live and browser"`), and that the default suite stays offline. Update the `vlm.py` module-map row to the real behavior.

- [ ] **Step 5: Run the full default suite**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_vlm_live.py README.md
git commit -m "test: add opt-in live VLM test and document M2"
```

---

## Self-Review

- **Spec coverage:** config/env/model-from-env ↔ Task 1 (§7.1); image/prompt/parse/fallback ↔ Task 2 (§7.3, §7.4, §7.5); client flow, error classification, repair, protocol errors ↔ Task 3 (§7.2, §7.5, §7.6); offline tests ↔ Task 3; live test + docs ↔ Task 4 (§7.7); markers/offline gate ↔ Task 1 + each task's suite step (§7.7); `.env` loading ↔ Task 1 (§5/§7.1).
- **Type consistency:** `VLMConfig`, `load_vlm_config`, `VLMError`, `encode_image_data_url`, `build_messages`, `parse_decision`, `fallback_decision`, `VLMClient(config, transport)`, `decide`, `_post_json`, `_extract_content` are named identically wherever produced/consumed. Transport signature `(str, dict, dict, float) -> dict` is consistent across producer and all tests.
- **Review Focus:** items 1–5 pinned in Tasks 3, 1+3, 1, 2–3, 3 respectively.
- **Deferred (design §9):** exact repair wording, numeric defaults, usage logging.
