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
