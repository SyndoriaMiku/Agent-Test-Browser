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

def test_timestamp_comes_from_injected_clock(tmp_path, clock):
    clock.advance(2.5)
    log = SessionLogger(tmp_path / "s.jsonl", clock=clock)
    log.log_observation(OBS)
    log.close()
    assert _read(tmp_path / "s.jsonl")[0]["timestamp"] == 2.5

def test_log_action_serializes_rejection_reason(tmp_path, clock):
    from game_agent.validation import RejectedReason, ValidationResult
    log = SessionLogger(tmp_path / "s.jsonl", clock=clock)
    action = Decision.model_validate(
        {"state":"s","reason":"r","confidence":0.9,"action":{"type":"click","x":1,"y":1}}).action
    log.log_action(OBS, action,
                   result=ValidationResult(accepted=False, reason=RejectedReason.OUT_OF_BOUNDS))
    log.close()
    row = _read(tmp_path / "s.jsonl")[0]
    assert row["accepted"] is False
    assert row["reason"] == "out_of_bounds"

def test_log_action_without_result_has_no_accepted_key(tmp_path, clock):
    log = SessionLogger(tmp_path / "s.jsonl", clock=clock)
    action = Decision.model_validate(
        {"state":"s","reason":"r","confidence":0.9,"action":{"type":"click","x":1,"y":1}}).action
    log.log_action(OBS, action)
    log.close()
    row = _read(tmp_path / "s.jsonl")[0]
    assert row["event"] == "action"
    assert "accepted" not in row
    assert "reason" not in row

def test_creates_missing_parent_directory(tmp_path, clock):
    log = SessionLogger(tmp_path / "logs" / "s.jsonl", clock=clock)
    log.log_observation(OBS)
    log.close()
    assert (tmp_path / "logs" / "s.jsonl").exists()

def test_default_clock_is_monotonic(tmp_path):
    log = SessionLogger(tmp_path / "s.jsonl")
    log.log_observation(OBS)
    log.close()
    row = _read(tmp_path / "s.jsonl")[0]
    assert isinstance(row["timestamp"], float)
