import base64
import urllib.error
import pytest
from game_agent import vlm
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

# --- extra coverage beyond the brief -----------------------------------------

async def test_empty_key_counts_as_missing(tmp_path):
    # A copied .env.example yields NVIDIA_API_KEY="" -> must be treated as missing.
    with pytest.raises(VLMError):
        await VLMClient(VLMConfig(model="m", api_key=""), transport=lambda *a: reply(VALID)).decide(obs(tmp_path), "g", 1)

async def test_empty_model_counts_as_missing(tmp_path):
    with pytest.raises(VLMError):
        await VLMClient(VLMConfig(model="", api_key="k"), transport=lambda *a: reply(VALID)).decide(obs(tmp_path), "g", 1)

async def test_error_messages_never_contain_the_key(tmp_path):
    def boom(*a): raise RuntimeError("network down")
    with pytest.raises(VLMError) as exc:
        await VLMClient(cfg(), transport=boom).decide(obs(tmp_path), "g", 1)
    assert "secret-key" not in str(exc.value)

async def test_config_resolution_uses_env_when_no_config_given(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    monkeypatch.setenv("NVIDIA_MODEL", "env-model")
    seen = {}
    def fake(url, headers, payload, timeout_s):
        seen.update(url=url, headers=headers); return reply(VALID)
    await VLMClient(transport=fake).decide(obs(tmp_path), "g", 1)
    assert seen["headers"]["Authorization"] == "Bearer env-key"
    assert seen["url"].endswith("/chat/completions")

async def test_trailing_slash_in_base_url_is_normalised(tmp_path):
    seen = {}
    def fake(url, headers, payload, timeout_s):
        seen["url"] = url; return reply(VALID)
    c = VLMConfig(base_url="https://example.test/v1/", model="m", api_key="k")
    await VLMClient(c, transport=fake).decide(obs(tmp_path), "g", 1)
    assert seen["url"] == "https://example.test/v1/chat/completions"

async def test_transport_is_called_off_the_event_loop(tmp_path):
    import threading
    main = threading.get_ident()
    seen = {}
    def fake(url, headers, payload, timeout_s):
        seen["thread"] = threading.get_ident(); return reply(VALID)
    await VLMClient(cfg(), transport=fake).decide(obs(tmp_path), "g", 1)
    assert seen["thread"] != main

async def test_repair_request_reuses_the_image_and_adds_assistant_and_user_turns(tmp_path):
    seen = []
    def fake(url, headers, payload, timeout_s):
        seen.append(payload["messages"]); return reply("garbage" if len(seen) == 1 else VALID)
    await VLMClient(cfg(), transport=fake).decide(obs(tmp_path), "g", 1)
    first, second = seen
    assert len(second) == len(first) + 2
    assert second[-2]["role"] == "assistant" and second[-2]["content"] == "garbage"
    assert second[-1]["role"] == "user" and "JSON" in second[-1]["content"]
    images = [p for m in second for p in (m["content"] if isinstance(m["content"], list) else [])
              if p.get("type") == "image_url"]
    assert len(images) == 1  # the screenshot is sent once, not twice

async def test_fallback_decision_carries_the_observation_id(tmp_path):
    def fake(url, headers, payload, timeout_s):
        return reply("still garbage")
    d = await VLMClient(cfg(), transport=fake).decide(obs(tmp_path), "g", 1)
    assert d.observation_id == "obs-1"

async def test_model_supplied_observation_id_is_overwritten(tmp_path):
    model_json = '{"state":"m","reason":"r","confidence":0.9,"observation_id":"model-id",' \
                 '"action":{"type":"wait","duration_ms":100}}'
    def fake(url, headers, payload, timeout_s):
        return reply(model_json)
    d = await VLMClient(cfg(), transport=fake).decide(obs(tmp_path), "g", 1)
    assert d.observation_id == "obs-1"

async def test_payload_carries_model_tokens_and_temperature(tmp_path):
    seen = {}
    def fake(url, headers, payload, timeout_s):
        seen.update(payload=payload, timeout_s=timeout_s); return reply(VALID)
    c = VLMConfig(model="m", api_key="k", max_tokens=123, temperature=0.25, timeout_s=9.5)
    await VLMClient(c, transport=fake).decide(obs(tmp_path), "g", 1)
    assert seen["payload"]["max_tokens"] == 123
    assert seen["payload"]["temperature"] == 0.25
    assert seen["timeout_s"] == 9.5

async def test_missing_screenshot_file_raises(tmp_path):
    o = Observation(observation_id="obs-1", screenshot_path=str(tmp_path / "nope.png"),
                    viewport_width=1, viewport_height=1, device_scale_factor=1.0, timestamp=0.0)
    with pytest.raises(VLMError):
        await VLMClient(cfg(), transport=lambda *a: reply(VALID)).decide(o, "g", 1)

async def test_protocol_error_for_null_content_raises(tmp_path):
    with pytest.raises(VLMError):
        await VLMClient(cfg(), transport=lambda *a: {"choices": [{"message": {"content": None}}]}).decide(obs(tmp_path), "g", 1)

# --- default transport (_post_json) ------------------------------------------

class FakeResponse:
    def __init__(self, body, status=200):
        self._body = body
        self.status = status
    def read(self):
        return self._body
    def getcode(self):
        return self.status
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False

def test_post_json_posts_json_and_returns_the_body(monkeypatch):
    captured = {}
    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data
        captured["timeout"] = timeout
        return FakeResponse(b'{"choices": []}')
    monkeypatch.setattr(vlm.urllib.request, "urlopen", fake_urlopen)
    out = vlm._post_json("https://x.test/v1/chat/completions",
                         {"Authorization": "Bearer secret-key"},
                         {"model": "m"}, 12.5)
    assert out == {"choices": []}
    assert captured["timeout"] == 12.5
    assert captured["body"] == b'{"model": "m"}'
    assert captured["headers"]["Authorization"] == "Bearer secret-key"

def test_post_json_raises_on_http_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 500, "Server Error", {}, None)
    monkeypatch.setattr(vlm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(VLMError) as exc:
        vlm._post_json("https://x.test/v1/chat/completions", {}, {}, 5.0)
    assert "500" in str(exc.value)

def test_post_json_raises_on_network_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(vlm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(VLMError):
        vlm._post_json("https://x.test/v1/chat/completions", {}, {}, 5.0)

def test_post_json_raises_on_non_json_body(monkeypatch):
    monkeypatch.setattr(vlm.urllib.request, "urlopen",
                        lambda request, timeout=None: FakeResponse(b"<html>not json</html>"))
    with pytest.raises(VLMError):
        vlm._post_json("https://x.test/v1/chat/completions", {}, {}, 5.0)
