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
    async def screenshot(self):
        if self.fail_screenshot:
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
