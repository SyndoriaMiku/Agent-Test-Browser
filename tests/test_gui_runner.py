"""Tests for the GUI background runner: one session on its own thread and
the event queue it publishes to.

All offline -- the browser and the model are fakes, the clock and the sleep
are injected, so the tests never wait for real time.
"""

import asyncio
import json
import queue
import threading
import time
from pathlib import Path

import game_agent.gui_runner as gui_runner
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


class UsageVLM(FakeVLM):
    """A model that reports its usage, so the log carries model records."""

    model = "fake-vlm"
    last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_the_queue_carries_the_same_records_as_the_jsonl_log(tmp_path, clock):
    q = queue.Queue()
    log_path = tmp_path / "g6.jsonl"
    runner = GuiRunner(q,
        controller_factory=lambda url, settings: FakeController(tmp_path, [b"A", b"B"]),
        vlm_factory=lambda s: UsageVLM([dec({"type": "click", "x": 1, "y": 1}),
                                         dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(log_path)})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    session_events = {"observation", "decision", "action", "verify", "model"}
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    written = [row for row in rows if row["event"] in session_events]
    logged = [e for e in _events(q) if e["event"] in session_events]
    assert written, "the session wrote no records"
    assert "model" in [row["event"] for row in written], "the model record is missing"
    assert logged == written


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
    end = next(e for e in events if e["event"] == "session_end")
    assert end["cycles"] == 1
    assert "done" in end["state"].lower()
    assert end["reason"]
    assert events[-1] == {"event": "status", "status": "finished"}


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


# --- shutdown ----------------------------------------------------------------


def test_shutdown_closes_the_browser_of_a_finished_run(tmp_path, clock):
    """A finished run leaves its browser open; shutdown closes it."""
    q = queue.Queue()
    controllers = []
    def cf(url, settings):
        c = FakeController(tmp_path, [b"A"]); controllers.append(c); return c
    runner = GuiRunner(q, controller_factory=cf,
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g14.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    assert controllers[0].closed is False
    runner.shutdown(timeout=15)
    assert controllers[0].closed is True
    assert runner.is_running is False


def test_shutdown_stops_the_run_and_closes_the_browser(tmp_path, clock):
    q = queue.Queue()
    controllers = []
    def cf(url, settings):
        c = FakeController(tmp_path, [b"A", b"B", b"C"]); controllers.append(c); return c
    runner = GuiRunner(q, controller_factory=cf,
        vlm_factory=lambda s: FakeVLM([dec({"type": "click", "x": 1, "y": 1})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g15.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.shutdown(timeout=15)
    assert runner.is_running is False
    assert controllers[0].closed is True


def test_shutdown_can_be_called_twice(tmp_path, clock):
    """The close happens exactly once, however often shutdown is called."""
    q = queue.Queue()
    controllers = []
    def cf(url, settings):
        c = FakeController(tmp_path, [b"A"]); controllers.append(c); return c
    runner = GuiRunner(q, controller_factory=cf,
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g16.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    runner.shutdown(timeout=15)
    runner.shutdown(timeout=15)
    assert controllers[0].closed is True
    stopped = [e for e in _events(q)
               if e["event"] == "status" and e["status"] == "stopped"]
    assert len(stopped) == 1


def test_shutdown_of_a_runner_that_never_started_returns(tmp_path, clock):
    q = queue.Queue()
    runner = GuiRunner(q,
        controller_factory=lambda url, settings: FakeController(tmp_path, [b"A"]),
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g17.jsonl")})
    runner.shutdown(timeout=5)
    assert runner.is_running is False


def test_shutdown_returns_even_when_the_browser_cannot_open(tmp_path, clock):
    """A browser that never opens must not make shutdown wait forever."""
    entered = threading.Event()
    release = threading.Event()

    class StuckController(FakeController):
        async def open(self):
            self.opened = True
            entered.set()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, release.wait)

    q = queue.Queue()
    runner = GuiRunner(q,
        controller_factory=lambda url, settings: StuckController(tmp_path, [b"A"]),
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g18.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    assert entered.wait(10), "the browser never started to open"
    started = time.monotonic()
    try:
        runner.shutdown(timeout=1)
    finally:
        release.set()
    assert time.monotonic() - started < 10, "shutdown waited for the stuck browser"
    runner.join(timeout=15)


# --- the browser profile -------------------------------------------------------


def test_each_runner_gets_its_own_browser_profile(monkeypatch):
    """Chromium locks a persistent profile exclusively.

    A restart launches the new browser while the previous one is still
    closing, so two runners must never share a profile directory.
    """
    built = []

    class RecordingController:
        """A controller that only remembers the options it was built with."""

        def __init__(self, settings, options):
            built.append(options)

    monkeypatch.setattr(gui_runner, "BrowserController", RecordingController)
    first = GuiRunner(queue.Queue())
    second = GuiRunner(queue.Queue())
    first._controller_factory("about:blank", Settings())
    first._controller_factory("about:blank", Settings())
    second._controller_factory("about:blank", Settings())

    assert len(built) == 3
    assert built[0].user_data_dir == built[1].user_data_dir, (
        "a runner keeps the profile of its own run"
    )
    assert built[0].user_data_dir != built[2].user_data_dir, (
        "two runners must not share a browser profile"
    )
    assert Path(built[0].user_data_dir).parent == Path(".browser-profile")
    assert built[0].headless is False


# --- the stop/close chain ----------------------------------------------------


def test_a_browser_that_could_not_be_opened_leaves_the_join_bounded(tmp_path, clock):
    """Without a browser there is nothing to close, so join must return."""
    q = queue.Queue()
    def cf(url, settings):
        raise RuntimeError("no browser")
    runner = GuiRunner(q, controller_factory=cf,
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g19.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    errors = [e for e in _events(q) if e["event"] == "error"]
    assert errors and "no browser" in errors[0]["message"]
    runner.stop()
    returned = threading.Event()
    threading.Thread(target=lambda: (runner.join(), returned.set()), daemon=True).start()
    assert returned.wait(10), "join() did not return after the browser failed to open"


def test_a_stop_that_lands_in_the_teardown_still_closes_the_browser(tmp_path, clock, monkeypatch):
    """A stop that arrives while the run tears down still closes the browser."""
    in_teardown = threading.Event()
    leave_teardown = threading.Event()

    class SlowLogger(gui_runner._QueueLogger):
        """A logger whose close holds the run in its teardown."""

        def close(self):
            in_teardown.set()
            leave_teardown.wait(10)
            super().close()

    monkeypatch.setattr(gui_runner, "_QueueLogger", SlowLogger)
    q = queue.Queue()
    controllers = []
    def cf(url, settings):
        c = FakeController(tmp_path, [b"A"]); controllers.append(c); return c
    runner = GuiRunner(q, controller_factory=cf,
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g20.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    assert in_teardown.wait(10), "the run never reached its teardown"
    runner.stop()
    leave_teardown.set()
    runner.join(timeout=15)
    assert controllers[0].closed is True


# --- the status vocabulary ----------------------------------------------------


def test_a_finished_run_reports_the_finished_status(tmp_path, clock):
    q = queue.Queue()
    runner = GuiRunner(q,
        controller_factory=lambda url, settings: FakeController(tmp_path, [b"A"]),
        vlm_factory=lambda s: FakeVLM([dec({"type": "done", "summary": "ok"})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g21.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.join(timeout=15)
    statuses = [e["status"] for e in _events(q) if e["event"] == "status"]
    assert statuses == ["running", "finished"]


def test_a_stopped_run_reports_the_stopped_status(tmp_path, clock):
    """The stop lands before the session can end on its own."""
    release = threading.Event()

    class WaitingController(FakeController):
        """A controller whose first capture waits for the stop."""

        async def screenshot(self):
            release.wait(10)
            return await super().screenshot()

    q = queue.Queue()
    runner = GuiRunner(q,
        controller_factory=lambda url, settings: WaitingController(tmp_path, [b"A", b"B", b"C"]),
        vlm_factory=lambda s: FakeVLM([dec({"type": "click", "x": 1, "y": 1})]),
        session_kwargs={"clock": clock, "sleep": make_sleep(clock), "logger_out": str(tmp_path / "g22.jsonl")})
    runner.start("about:blank", "goal", Settings(), grid_step=None)
    runner.stop()
    release.set()
    runner.join(timeout=15)
    statuses = [e["status"] for e in _events(q) if e["event"] == "status"]
    assert "stopped" in statuses
    assert "finished" not in statuses
