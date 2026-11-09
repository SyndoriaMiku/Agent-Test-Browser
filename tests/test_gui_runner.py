"""Tests for the GUI background runner: one session on its own thread and
the event queue it publishes to.

All offline -- the browser and the model are fakes, the clock and the sleep
are injected, so the tests never wait for real time.
"""

import json
import queue

from game_agent.config import Settings
from game_agent.gui_runner import GuiRunner
from tests.test_session import FakeController, FakeVLM, dec, make_sleep


def _events(q):
    """Drain *q* and return the events it holds."""
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


# --- tests required by the task brief --------------------------------------


def test_runner_runs_and_emits_events(tmp_path, clock):
    q = queue.Queue()
    frames = [b"A", b"B"]
    decisions = [dec({"type": "click", "x": 1, "y": 1}), dec({"type": "done", "summary": "ok"})]
    runner = GuiRunner(q,
        controller_factory=lambda url, settings: FakeController(tmp_path, frames),
        vlm_factory=lambda settings: FakeVLM(decisions),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    kinds = [e["event"] for e in _events(q)]
    assert "session_end" in kinds
    assert "observation" in kinds and "decision" in kinds and "action" in kinds


def test_stop_closes_browser(tmp_path, clock):
    q = queue.Queue()
    controllers = []
    def cf(url, settings):
        c = FakeController(tmp_path, [b"A", b"B", b"C"]); controllers.append(c); return c
    runner = GuiRunner(q, controller_factory=cf,
        vlm_factory=lambda s: FakeVLM([dec({"type": "click", "x": 1, "y": 1})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g2.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.stop()
    runner.join(timeout=15)
    assert controllers and controllers[0].closed is True


# --- lifecycle ---------------------------------------------------------------


def test_runner_opens_the_browser_and_navigates_to_the_url(tmp_path, clock):
    q = queue.Queue()
    controllers = []
    def cf(url, settings):
        c = FakeController(tmp_path, [b"A"]); controllers.append(c); return c
    runner = GuiRunner(q, controller_factory=cf,
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g3.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    assert controllers[0].opened is True
    assert controllers[0].url == "about:blank"


def test_a_finished_run_leaves_the_browser_open(tmp_path, clock):
    q = queue.Queue()
    controllers = []
    def cf(url, settings):
        c = FakeController(tmp_path, [b"A"]); controllers.append(c); return c
    runner = GuiRunner(q, controller_factory=cf,
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g4.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    assert controllers[0].closed is False
    assert runner.is_running is False


def test_a_stop_after_the_run_closes_the_browser_it_left_open(tmp_path, clock):
    q = queue.Queue()
    controllers = []
    def cf(url, settings):
        c = FakeController(tmp_path, [b"A"]); controllers.append(c); return c
    runner = GuiRunner(q, controller_factory=cf,
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g10.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    assert controllers[0].closed is False
    runner.stop()
    runner.join(timeout=15)
    assert controllers[0].closed is True


def test_a_runner_drives_a_single_run(tmp_path, clock):
    q = queue.Queue()
    runner = GuiRunner(q,
        controller_factory=lambda url, settings: FakeController(tmp_path, [b"A"]),
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g11.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    try:
        runner.start("about:blank", "goal", Settings(), grid_step=None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a runner must not start twice")


def test_a_failing_browser_is_reported_as_an_error(tmp_path, clock):
    class BrokenController(FakeController):
        async def open(self):
            raise RuntimeError("browser unavailable")

    q = queue.Queue()
    runner = GuiRunner(q,
        controller_factory=lambda url, settings: BrokenController(tmp_path, [b"A"]),
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g5.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    errors = [e for e in _events(q) if e["event"] == "error"]
    assert errors and "browser unavailable" in errors[0]["message"]


# --- queue protocol ----------------------------------------------------------


def test_the_queue_carries_the_same_records_as_the_jsonl_log(tmp_path, clock):
    q = queue.Queue()
    log_path = tmp_path / "g6.jsonl"
    runner = GuiRunner(q,
        controller_factory=lambda url, settings: FakeController(tmp_path, [b"A", b"B"]),
        vlm_factory=lambda s: FakeVLM([dec({"type": "click", "x": 1, "y": 1}),
                                        dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(log_path)})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    logged = [e for e in _events(q)
              if e["event"] in {"observation", "decision", "action", "verify"}]
    assert rows, "the session wrote no records"
    assert logged == rows


def test_the_run_reports_its_start_and_its_end(tmp_path, clock):
    q = queue.Queue()
    runner = GuiRunner(q,
        controller_factory=lambda url, settings: FakeController(tmp_path, [b"A"]),
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g7.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    events = _events(q)
    assert events[0] == {"event": "status", "status": "running"}
    end = events[-1]
    assert end["event"] == "session_end"
    assert end["cycles"] == 1
    assert "done" in end["state"].lower()
    assert end["reason"]


# --- pause / resume ----------------------------------------------------------


def test_pause_holds_the_session_until_it_is_resumed(tmp_path, clock):
    q = queue.Queue()
    gate_sleeps = []
    runner = None

    async def sleep(seconds):
        clock.advance(seconds)
        if seconds == 0.1:               # the session's pause gate
            gate_sleeps.append(seconds)
            runner.resume()

    class PausingVLM(FakeVLM):
        async def decide(self, observation, goal, remaining_actions):
            runner.pause()
            return await super().decide(observation, goal, remaining_actions)

    runner = GuiRunner(q,
        controller_factory=lambda url, settings: FakeController(tmp_path, [b"A", b"B", b"C"]),
        vlm_factory=lambda s: PausingVLM([dec({"type": "click", "x": 1, "y": 1}),
                                           dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": sleep, "logger_out": str(tmp_path / "g8.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    events = _events(q)
    assert gate_sleeps, "the session never waited at the pause gate"
    statuses = [e.get("status") for e in events if e["event"] == "status"]
    assert "paused" in statuses
    assert "session_end" in [e["event"] for e in events]


def test_pause_and_resume_do_nothing_without_a_session(tmp_path, clock):
    q = queue.Queue()
    runner = GuiRunner(q,
        controller_factory=lambda url, settings: FakeController(tmp_path, [b"A"]),
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g9.jsonl")})
    runner.pause()
    runner.resume()
    assert _events(q) == []
