import json

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

# --- extra coverage beyond the brief ------------------------------------------

def _obs(tmp_path):
    p = tmp_path / "s.png"; p.write_bytes(b"\x89PNG\r\n\x1a\n")
    from game_agent.observation import Observation
    return Observation("o", str(p), 10, 10, 1.0, 0.0)

async def test_usage_and_model_start_empty(tmp_path):
    client = VLMClient(VLMConfig(model="m", api_key="k"),
                       transport=lambda *a: _resp(VALID))
    assert client.model is None
    assert client.last_usage is None

async def test_partial_usage_is_tolerated(tmp_path):
    # The endpoint may report only some of the counters; the missing ones
    # count as zero and the call still succeeds.
    client = VLMClient(VLMConfig(model="m", api_key="k"),
                       transport=lambda *a: _resp(VALID, {"prompt_tokens": 5}))
    await client.decide(_obs(tmp_path), "goal", 1)
    assert client.last_usage == {"prompt_tokens": 5, "completion_tokens": 0,
                                  "total_tokens": 0}

async def test_usage_is_not_accumulated_across_decides(tmp_path):
    client = VLMClient(VLMConfig(model="m", api_key="k"),
                       transport=lambda *a: _resp(VALID, {"prompt_tokens": 4,
                                                         "completion_tokens": 1,
                                                         "total_tokens": 5}))
    await client.decide(_obs(tmp_path), "goal", 1)
    await client.decide(_obs(tmp_path), "goal", 1)
    assert client.last_usage == {"prompt_tokens": 4, "completion_tokens": 1,
                                  "total_tokens": 5}

async def test_usage_is_reset_when_the_next_call_reports_none(tmp_path):
    seq = [_resp(VALID, {"prompt_tokens": 4, "completion_tokens": 1,
                         "total_tokens": 5}), _resp(VALID)]
    calls = {"i": 0}
    def fake(url, headers, payload, t):
        r = seq[calls["i"]]; calls["i"] += 1; return r
    client = VLMClient(VLMConfig(model="m", api_key="k"), transport=fake)
    await client.decide(_obs(tmp_path), "goal", 1)
    assert client.last_usage is not None
    await client.decide(_obs(tmp_path), "goal", 1)
    assert client.last_usage is None

async def test_model_is_the_resolved_config_model(tmp_path):
    client = VLMClient(VLMConfig(model="resolved-model", api_key="k"),
                       transport=lambda *a: _resp(VALID))
    await client.decide(_obs(tmp_path), "goal", 1)
    assert client.model == "resolved-model"


# --- the session reports the model event --------------------------------------

class _FakeController:
    """A controller that always returns the same frame."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.shots = 0

    async def screenshot(self):
        from game_agent.observation import Observation
        self.shots += 1
        p = self.tmp / f"shot-{self.shots}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        return Observation("obs-1", str(p), 10, 10, 1.0, 0.0)

    async def execute(self, action):
        return None

    async def release_all(self):
        return None


class _FakeVLM:
    """A model that reports a decision and (optionally) its usage."""

    def __init__(self, usage, model="some-model"):
        self.last_usage = usage
        self.model = model

    async def decide(self, observation, goal, remaining_actions):
        from game_agent.schema import Decision
        return Decision.model_validate(
            {"state": "s", "reason": "r", "confidence": 0.9,
             "action": {"type": "done", "summary": "ok"}}
        )


async def _run_session(tmp_path, clock, vlm):
    from game_agent.config import Settings
    from game_agent.logging import SessionLogger
    from game_agent.session import Session
    log = SessionLogger(tmp_path / "session.jsonl", clock=clock)
    session = Session(_FakeController(tmp_path), vlm, Settings(), logger=log,
                      grid_step=None, clock=clock)
    await session.run("goal", max_cycles=2)
    log.close()
    rows = [json.loads(line) for line in
            (tmp_path / "session.jsonl").read_text(encoding="utf-8").splitlines()]
    return rows


async def test_session_logs_a_model_event_when_usage_is_reported(tmp_path, clock):
    usage = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    rows = await _run_session(tmp_path, clock, _FakeVLM(usage))
    model_rows = [row for row in rows if row["event"] == "model"]
    assert len(model_rows) == 1
    row = model_rows[0]
    assert row["observation_id"] == "obs-1"
    assert row["model"] == "some-model"
    assert row["prompt_tokens"] == 5
    assert row["completion_tokens"] == 7
    assert row["total_tokens"] == 12


async def test_session_logs_no_model_event_without_usage(tmp_path, clock):
    rows = await _run_session(tmp_path, clock, _FakeVLM(None))
    assert [row["event"] for row in rows] == ["observation", "decision"]
