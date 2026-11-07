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
