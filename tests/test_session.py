import hashlib
import io

import pytest
from PIL import Image

from game_agent.config import Settings
from game_agent.controller import ControllerError
from game_agent.observation import Observation
from game_agent.schema import Decision
from game_agent.session import Session, SessionResult, _image_changed
from game_agent.state import State
from game_agent.vlm import VLMError


def frame_png(marker: bytes) -> bytes:
    """Render *marker* as a deterministic PNG frame.

    Screenshots are PNG files and the coordinate-grid overlay reads them
    with Pillow, so a frame has to be a real image.  The marker's hash
    picks the colour: different markers give different bytes (the screen
    changed) and the same marker gives the same bytes (the screen did not).
    """
    digest = hashlib.sha256(marker).digest()
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (digest[0], digest[1], digest[2])).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeController:
    def __init__(self, tmp_path, frames):
        self.tmp, self.frames, self.i = tmp_path, list(frames), 0
        self.executed, self.release_calls, self.fail_execute = [], 0, set()
        self.fail_screenshot = False
        self.fail_screenshot_on = set()   # 1-based screenshot call numbers that raise
        self.screenshot_calls = 0
        self.opened, self.closed, self.url = False, False, None
    async def open(self):
        self.opened = True
    async def goto(self, url):
        self.url = url
    async def close(self):
        self.closed = True
    async def screenshot(self):
        self.screenshot_calls += 1
        if self.fail_screenshot or self.screenshot_calls in self.fail_screenshot_on:
            raise ControllerError("screenshot failed")
        data = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        p = self.tmp / f"shot-{self.i}.png"
        p.write_bytes(frame_png(data))
        return Observation(observation_id=f"obs-{self.i}", screenshot_path=str(p),
                           viewport_width=1280, viewport_height=720,
                           device_scale_factor=1.0, timestamp=float(self.i))
    async def execute(self, action):
        if action.type in self.fail_execute:
            raise ControllerError("execute failed")
        self.executed.append(action)
    async def release_all(self):
        self.release_calls += 1

class FakeVLM:
    def __init__(self, decisions):
        self.decisions, self.calls, self.seen = list(decisions), 0, []
    async def decide(self, observation, goal, remaining_actions):
        self.seen.append(observation.screenshot_path)
        d = self.decisions[min(self.calls, len(self.decisions) - 1)]
        self.calls += 1
        if isinstance(d, Exception):
            raise d
        return Decision.model_validate(d)

def dec(action, conf=0.9):
    return {"state": "s", "reason": "r", "confidence": conf, "action": action}

def make_sleep(clock):
    async def _sleep(seconds):
        clock.advance(seconds)
    return _sleep

async def test_happy_path_reaches_done(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B"])   # second shot differs -> changed
    vlm = FakeVLM([dec({"type": "click", "x": 10, "y": 10}),
                   dec({"type": "done", "summary": "ok"})])
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=10)
    assert result.state == State.DONE
    assert [a.type for a in ctrl.executed] == ["click"]
    assert ctrl.release_calls >= 1

async def test_second_cycle_not_rate_limited(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B", b"C"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1}),
                   dec({"type": "click", "x": 2, "y": 2}),
                   dec({"type": "done", "summary": "ok"})])
    session = Session(ctrl, vlm, Settings(min_action_interval_ms=200), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=10)
    assert result.state == State.DONE                 # not NEEDS_USER(RATE_LIMITED)
    assert vlm.calls == 3

async def test_grid_overlay_path_is_sent_to_vlm(tmp_path, clock):
    import pathlib
    ctrl = FakeController(tmp_path, [b"A", b"B"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1}), dec({"type": "done", "summary": "ok"})])
    session = Session(ctrl, vlm, Settings(), grid_step=50, clock=clock, sleep=make_sleep(clock))
    await session.run("goal", max_cycles=5)
    assert vlm.seen[0].endswith(".grid.png")
    assert pathlib.Path(vlm.seen[0]).exists()

async def test_release_all_called_on_done(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B"])
    vlm = FakeVLM([dec({"type": "done", "summary": "ok"})])
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    await session.run("goal", max_cycles=5)
    assert ctrl.release_calls == 1

async def test_max_cycles_stops_the_loop(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B", b"C"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=2)
    assert result.state == State.STOPPED
    assert result.cycles == 2
    assert ctrl.release_calls == 1

async def test_image_changed_compares_bytes_and_treats_missing_as_changed(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(frame_png(b"A"))
    second.write_bytes(frame_png(b"A"))
    assert _image_changed(str(first), str(second)) is False
    second.write_bytes(frame_png(b"B"))
    assert _image_changed(str(first), str(second)) is True
    assert _image_changed(str(tmp_path / "gone.png"), str(second)) is True
    assert _image_changed(None, str(second)) is True

async def test_unchanged_image_hits_failure_cap(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])          # same bytes every shot -> never changes
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=2,
                      repeat_limit=99, clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=10)
    assert result.state == State.NEEDS_USER
    assert ctrl.release_calls == 1

async def test_wait_action_is_exempt_from_unchanged_failure(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])          # never changes
    vlm = FakeVLM([dec({"type": "wait", "duration_ms": 10}),
                   dec({"type": "done", "summary": "ok"})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=1,
                      repeat_limit=99, clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.DONE                 # wait does not false-fail

async def test_repeated_action_hits_repeat_limit(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B"])    # changes each time, but same action repeats
    vlm = FakeVLM([dec({"type": "click", "x": 5, "y": 5})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=99,
                      repeat_limit=2, clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=10)
    assert result.state == State.NEEDS_USER
    assert "repeat" in result.reason.lower()

async def test_budget_exhaustion_stops(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B", b"C", b"D"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    session = Session(ctrl, vlm, Settings(max_actions_per_session=2),
                      max_consecutive_failures=99, repeat_limit=99, clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=10)
    assert result.state == State.STOPPED
    assert len(ctrl.executed) <= 2

async def test_max_cycles_stops(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B", b"C"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=99, repeat_limit=99,
                      clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=2)
    assert result.state == State.STOPPED
    assert result.cycles <= 2

async def test_stop_ends_loop(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B", b"C"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=99, repeat_limit=99,
                      clock=clock, sleep=make_sleep(clock))
    session.stop()
    result = await session.run("goal", max_cycles=10)
    assert result.state == State.STOPPED
    assert ctrl.release_calls == 1
    assert ctrl.executed == []

async def test_vlm_error_needs_user(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    vlm = FakeVLM([VLMError("no key")])
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.NEEDS_USER
    assert ctrl.release_calls == 1

async def test_execute_error_recovers_or_caps(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    ctrl.fail_execute = {"click"}
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=1, repeat_limit=99,
                      clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.NEEDS_USER
    assert ctrl.release_calls == 1

async def test_execute_error_recovers_before_the_cap(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    ctrl.fail_execute = {"click"}
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=2,
                      repeat_limit=99, clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.NEEDS_USER         # second failure hits the cap
    assert vlm.calls == 2                           # the first one recovered
    assert ctrl.release_calls == 1

async def test_screenshot_error_needs_user(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    ctrl.fail_screenshot = True
    vlm = FakeVLM([dec({"type": "done", "summary": "x"})])
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.NEEDS_USER
    assert ctrl.release_calls == 1

async def test_session_logs_observation_decision_action_and_verify(tmp_path, clock):
    import json
    from game_agent.logging import SessionLogger
    ctrl = FakeController(tmp_path, [b"A", b"B"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1}), dec({"type": "done", "summary": "ok"})])
    log = SessionLogger(tmp_path / "session.jsonl", clock=clock)
    session = Session(ctrl, vlm, Settings(), logger=log, clock=clock, sleep=make_sleep(clock))
    await session.run("goal", max_cycles=5)
    log.close()
    rows = [json.loads(line) for line in
            (tmp_path / "session.jsonl").read_text(encoding="utf-8").splitlines()]
    events = [row["event"] for row in rows]
    assert events[:4] == ["observation", "decision", "action", "verify"]
    assert events[-2:] == ["observation", "decision"]     # the done decision runs no action
    assert all(row["observation_id"] for row in rows)
    assert rows[3]["changed"] is True

async def test_low_confidence_decision_is_rejected_and_needs_user(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    vlm = FakeVLM([dec({"type": "click", "x": 10, "y": 10}, conf=0.1)])  # under the threshold
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.NEEDS_USER
    assert "rejected" in result.reason.lower()
    assert "low_confidence" in result.reason
    assert ctrl.executed == []                # a rejected decision never acts
    assert ctrl.release_calls == 1

async def test_out_of_bounds_decision_is_rejected_and_needs_user(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    vlm = FakeVLM([dec({"type": "click", "x": 5000, "y": 10})])  # outside the 1280x720 viewport
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.NEEDS_USER
    assert "rejected" in result.reason.lower()
    assert "out_of_bounds" in result.reason
    assert ctrl.executed == []
    assert ctrl.release_calls == 1

async def test_uncertain_decision_needs_user(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    vlm = FakeVLM([dec({"type": "uncertain", "reason": "screen unreadable"})])
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.NEEDS_USER
    assert "uncertain" in result.reason.lower()
    assert ctrl.executed == []                # an uncertain decision never acts
    assert ctrl.release_calls == 1

async def test_verification_screenshot_error_counts_as_a_failure(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    ctrl.fail_screenshot_on = {2}            # observe (1) works, verify (2) raises
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=1, repeat_limit=99,
                      clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.NEEDS_USER   # the recorded failure hit the cap
    assert "verification failed" in result.reason.lower()
    assert "failure cap reached" in result.reason.lower()
    assert [a.type for a in ctrl.executed] == ["click"]
    assert ctrl.release_calls == 1

async def test_verification_screenshot_error_recovers_before_the_cap(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B"])
    ctrl.fail_screenshot_on = {2, 4}         # both verification captures raise
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1}),
                   dec({"type": "click", "x": 2, "y": 2})])
    session = Session(ctrl, vlm, Settings(), max_consecutive_failures=2, repeat_limit=99,
                      clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.NEEDS_USER   # the second failure hits the cap
    assert vlm.calls == 2                    # the first failure was recorded and recovered
    assert len(ctrl.executed) == 2
    assert ctrl.release_calls == 1
