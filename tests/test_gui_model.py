"""Tests for the pure GUI display model: event formatting, preview paths,
token/cost tracking.  All offline -- no Tk, no network, no session."""

import json

import pytest

from game_agent.gui_model import UsageTracker, format_event, preview_path


# --- tests required by the task brief --------------------------------------


def test_format_event_known_types():
    s = format_event({"event": "decision", "observation_id": "obs-1",
                      "decision": {"state": "menu", "reason": "start visible",
                                   "action": {"type": "click", "x": 1, "y": 2}}})
    assert "decision" in s and "start visible" in s and "click" in s

def test_format_event_unknown_type_is_safe():
    assert isinstance(format_event({"event": "weird", "x": 1}), str)

def test_preview_path_prefers_grid(tmp_path):
    raw = tmp_path / "shot.png"; raw.write_bytes(b"x")
    (tmp_path / "shot.grid.png").write_bytes(b"y")
    assert preview_path(str(raw)).endswith("shot.grid.png")
    raw2 = tmp_path / "only.png"; raw2.write_bytes(b"x")
    assert preview_path(str(raw2)) == str(raw2)
    assert preview_path(None) is None

def test_usage_tracker_sums_and_costs():
    t = UsageTracker()
    t.add({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
    t.add({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    assert (t.prompt_tokens, t.completion_tokens, t.total_tokens) == (110, 55, 165)
    # Prices are USD per 1,000,000 tokens: 110 prompt @ $1/1M + 55 completion @ $2/1M.
    assert t.cost_usd(1.0, 2.0) == pytest.approx(110 / 1e6 * 1.0 + 55 / 1e6 * 2.0)
    assert t.cost_usd(None, None) is None


# --- every event kind renders readably (spec §7.8) --------------------------


def test_format_observation_shows_screenshot():
    s = format_event({"event": "observation", "observation_id": "obs-1",
                      "screenshot_path": r"C:\tmp\shots\shot-001.png"})
    assert "observation" in s and "obs-1" in s and "shot-001.png" in s

def test_format_action_accepted_means_validation_only():
    s = format_event({"event": "action", "observation_id": "obs-1",
                      "action": {"type": "click", "x": 3, "y": 4},
                      "accepted": True})
    assert "accepted (validation)" in s
    assert "success" not in s.lower() and "succeed" not in s.lower()

def test_format_action_rejected_shows_reason():
    s = format_event({"event": "action", "observation_id": "obs-1",
                      "action": {"type": "wait", "duration_ms": 10},
                      "accepted": False, "reason": "rate_limited"})
    assert "rejected" in s and "rate_limited" in s

def test_format_action_null_accepted_is_no_verdict():
    s = format_event({"event": "action", "observation_id": "obs-1",
                      "action": {"type": "wait", "duration_ms": 10},
                      "accepted": None})
    assert "accepted" not in s and "rejected" not in s

def test_format_action_without_validation_outcome():
    s = format_event({"event": "action", "observation_id": "obs-1",
                      "action": {"type": "press", "key": "Enter"}})
    assert "press" in s and "accepted" not in s and "rejected" not in s

def test_format_verify_reports_change_without_echoing_note():
    changed = format_event({"event": "verify", "observation_id": "obs-1",
                            "changed": True, "note": "screen changed"})
    assert "verify" in changed and "screen changed" in changed
    assert changed.count("screen changed") == 1
    extra = format_event({"event": "verify", "observation_id": "obs-1",
                          "changed": False, "note": "popup blocked the click"})
    assert "screen unchanged" in extra and "popup blocked the click" in extra

def test_format_model_shows_name_and_tokens():
    s = format_event({"event": "model", "observation_id": "obs-1",
                      "model": "some-model",
                      "prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12})
    assert "some-model" in s and "5" in s and "7" in s and "12" in s

def test_format_model_null_tokens_read_as_zero():
    s = format_event({"event": "model", "observation_id": "obs-1",
                      "model": None, "prompt_tokens": None,
                      "completion_tokens": None, "total_tokens": None})
    assert "0 in / 0 out / 0 total" in s

def test_format_runner_events():
    assert "running" in format_event({"event": "status", "status": "running"})
    end = format_event({"event": "session_end", "state": "DONE",
                        "cycles": 3, "reason": "goal reported complete"})
    assert "session_end" in end and "DONE" in end and "3" in end
    assert "goal reported complete" in end
    assert "boom" in format_event({"event": "error", "message": "boom"})

def test_format_event_never_crashes_on_missing_or_null_fields():
    for kind in ("observation", "decision", "action", "verify", "model",
                 "status", "session_end", "error", "weird"):
        assert isinstance(format_event({"event": kind}), str)
        assert isinstance(format_event({"event": kind, "decision": None,
                                        "action": None, "screenshot_path": None,
                                        "changed": None, "note": None}), str)

def test_format_event_handles_a_real_jsonl_record():
    row = json.loads('{"event": "action", "timestamp": 1.0, "observation_id": "obs-1",'
                     ' "action": {"type": "wait", "duration_ms": 10}}')
    s = format_event(row)
    assert "wait" in s and "obs-1" in s

@pytest.mark.parametrize("action,needle", [
    ({"type": "click", "x": 1, "y": 2}, "click @ (1, 2)"),
    ({"type": "click", "x": 1, "y": 2, "button": "right"}, "click @ (1, 2) right"),
    ({"type": "click", "x": 1, "y": 2, "click_count": 2}, "click @ (1, 2) x2"),
    ({"type": "click", "x": 1, "y": 2, "button": "middle", "click_count": 3},
     "click @ (1, 2) middle x3"),
    ({"type": "move", "x": 4, "y": 5}, "move @ (4, 5)"),
    ({"type": "press", "key": "Enter"}, "Enter"),
    ({"type": "hold", "key": "a", "duration_ms": 200}, "200ms"),
    ({"type": "release", "button": "left"}, "release"),
    ({"type": "drag", "start_x": 1, "start_y": 2, "end_x": 3, "end_y": 4}, "drag"),
    ({"type": "wheel", "delta_y": 120}, "120"),
    ({"type": "wait", "duration_ms": 500}, "500ms"),
    ({"type": "done", "summary": "level cleared"}, "level cleared"),
    ({"type": "uncertain", "reason": "too dark"}, "too dark"),
    ({"type": "hover", "x": 1}, "hover"),
])
def test_format_action_variants(action, needle):
    s = format_event({"event": "decision", "observation_id": "o",
                      "decision": {"state": "s", "reason": "r", "action": action}})
    assert needle in s


# --- preview path edge cases -------------------------------------------------


def test_preview_path_missing_file_still_returns_path(tmp_path):
    missing = tmp_path / "gone.png"
    assert preview_path(str(missing)) == str(missing)

def test_preview_path_empty_string_is_none():
    assert preview_path("") is None


# --- usage tracker tolerance -------------------------------------------------


def test_usage_tracker_tolerates_missing_keys_and_none():
    t = UsageTracker()
    t.add(None)
    t.add({})
    assert (t.prompt_tokens, t.completion_tokens, t.total_tokens) == (0, 0, 0)
    t.add({"prompt_tokens": 3})
    assert (t.prompt_tokens, t.completion_tokens, t.total_tokens) == (3, 0, 0)

def test_usage_tracker_cost_needs_both_prices():
    t = UsageTracker()
    t.add({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
    assert t.cost_usd(None, 2.0) is None
    assert t.cost_usd(1.0, None) is None
    # 1M prompt tokens @ $1/1M + 1M completion tokens @ $2/1M = $3.00.
    assert t.cost_usd(1.0, 2.0) == pytest.approx(3.0)
