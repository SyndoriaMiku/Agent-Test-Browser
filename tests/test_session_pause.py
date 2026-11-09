import json

import pytest
from game_agent.config import Settings
from game_agent.logging import SessionLogger
from game_agent.schema import Decision
from game_agent.session import Session
from game_agent.state import State

# reuse the FakeController/FakeVLM/dec/make_sleep helpers pattern from tests/test_session.py
from tests.test_session import FakeController, FakeVLM, dec, make_sleep  # type: ignore

async def test_pause_blocks_then_resume_continues(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B", b"C"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1}), dec({"type": "done", "summary": "ok"})])
    calls = {"n": 0}
    session_ref = {}
    async def resume_sleep(seconds):
        clock.advance(seconds)
        calls["n"] += 1
        if calls["n"] >= 2:            # gate entered >= twice while paused, then resume
            session_ref["s"].resume()
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=resume_sleep)
    session_ref["s"] = session
    session.pause()
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.DONE       # paused, then resumed, then ran to DONE
    assert calls["n"] >= 2                  # the pause gate actually looped

async def test_stop_beats_pause(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    vlm = FakeVLM([dec({"type": "done", "summary": "x"})])
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=make_sleep(clock))
    session.pause(); session.stop()
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.STOPPED and ctrl.executed == []

async def test_accessors(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A", b"B"])
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1}), dec({"type": "done", "summary": "ok"})])
    session = Session(ctrl, vlm, Settings(max_actions_per_session=5), clock=clock, sleep=make_sleep(clock))
    result = await session.run("goal", max_cycles=5)
    assert session.state == State.DONE and session.cycles == result.cycles
    assert session.remaining_actions == 5 - len(ctrl.executed)

async def test_resume_clears_the_pause(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    vlm = FakeVLM([dec({"type": "done", "summary": "ok"})])
    calls = {"n": 0}
    async def counting_sleep(seconds):
        calls["n"] += 1
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=counting_sleep)
    session.pause()
    session.resume()
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.DONE
    assert calls["n"] == 0                  # the gate never had to wait

async def test_pause_between_cycles_blocks_the_next_cycle(tmp_path, clock):
    events = []
    calls = {"n": 0}
    session_ref = {}

    class RecordingController(FakeController):
        async def screenshot(self):
            events.append("screenshot")
            return await super().screenshot()

    class PausingVLM(FakeVLM):
        async def decide(self, observation, goal, remaining_actions):
            decision = await super().decide(observation, goal, remaining_actions)
            if self.calls == 1:             # pause right after the first decision
                session_ref["s"].pause()
            return decision

    async def resume_sleep(seconds):
        clock.advance(seconds)
        calls["n"] += 1
        events.append("sleep")
        if calls["n"] >= 2:                 # the gate waited, then resume
            session_ref["s"].resume()

    ctrl = RecordingController(tmp_path, [b"A", b"B", b"C"])
    vlm = PausingVLM([dec({"type": "click", "x": 1, "y": 1}),
                      dec({"type": "done", "summary": "ok"})])
    session = Session(ctrl, vlm, Settings(), clock=clock, sleep=resume_sleep)
    session_ref["s"] = session
    result = await session.run("goal", max_cycles=5)
    assert result.state == State.DONE
    shots = [i for i, e in enumerate(events) if e == "screenshot"]
    # shot 1 = observe of cycle 1, shot 2 = verify of cycle 1, shot 3 = observe
    # of cycle 2 -- the gate has to sit between the verify and that observe.
    assert len(shots) >= 3
    assert events[shots[1] + 1] == "sleep"      # cycle 2 waited at the gate
    assert events[:shots[2]].count("sleep") >= 2

async def test_failed_action_is_logged(tmp_path, clock):
    ctrl = FakeController(tmp_path, [b"A"])
    ctrl.fail_execute = {"click"}
    vlm = FakeVLM([dec({"type": "click", "x": 1, "y": 1})])
    log = SessionLogger(tmp_path / "session.jsonl", clock=clock)
    session = Session(ctrl, vlm, Settings(), logger=log, max_consecutive_failures=1,
                      clock=clock, sleep=make_sleep(clock))
    await session.run("goal", max_cycles=5)
    log.close()
    rows = [json.loads(line) for line in
            (tmp_path / "session.jsonl").read_text(encoding="utf-8").splitlines()]
    actions = [row for row in rows if row["event"] == "action"]
    assert len(actions) == 1                       # the failed attempt is in the log
    assert actions[0]["action"]["type"] == "click"
    assert actions[0]["accepted"] is True          # validation had accepted it
