"""Tests for the Tk view: the restart order and the non-blocking Tk thread.

The tests build the real Tk window, so they need a display and carry the
``gui`` marker (the default suite deselects them).  The browser itself is
a fake: what matters here is the order -- the previous runner is shut down
first, and the next run starts only after that close is done.
"""

import gc
import threading
import time
import tkinter as tk

import pytest

import game_agent.gui as gui
from game_agent.gui import GameAgentGui


class FakeRunner:
    """A runner that records what the view asked of it."""

    def __init__(self, events):
        """Remember the queue and prepare the recording fields."""
        self.events = events
        self.started = None
        self.shutdown_timeouts = []
        self.release_shutdown = threading.Event()
        self.stops = 0

    def start(self, url, goal, settings, viewport=None, grid_step=None):
        """Record the arguments of the run."""
        self.started = (url, goal, settings, viewport, grid_step)

    def shutdown(self, timeout=None):
        """Record the shutdown and wait until the test releases it."""
        self.shutdown_timeouts.append(timeout)
        self.release_shutdown.wait(10)

    def stop(self):
        """Count the stop requests."""
        self.stops += 1

    def pause(self):
        """Ignore the pause request."""

    def resume(self):
        """Ignore the resume request."""

    @property
    def is_running(self):
        """Pretend the run is still going."""
        return True


class RunnerFactory:
    """Builds fake runners and keeps them for the assertions."""

    def __init__(self):
        """Start without a runner."""
        self.runners = []

    def __call__(self, events):
        """Build a fake runner for *events*."""
        runner = FakeRunner(events)
        self.runners.append(runner)
        return runner


def _make_app():
    """Build the real window, skipping the test when there is no display."""
    try:
        app = GameAgentGui()
    except tk.TclError as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no display available: {exc}")
    app.withdraw()
    return app


@pytest.fixture
def app():
    """The real window, torn down on the main thread.

    The Tk objects form reference cycles, so the interpreter would
    otherwise be collected much later -- possibly on the worker thread of
    another test, which breaks that test.
    """
    application = _make_app()
    yield application
    application.destroy()
    del application
    gc.collect()


def _pump_until(app, predicate, timeout=10.0):
    """Poll the view until *predicate* holds; fail when it never does."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app._poll()
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("the view never finished the pending start")


@pytest.mark.gui
def test_a_restart_waits_for_the_previous_browser_to_close(monkeypatch, app):
    """The next run starts only after the previous runner is shut down."""
    factory = RunnerFactory()
    monkeypatch.setattr(gui, "GuiRunner", factory)
    app._url_var.set("about:blank")
    app._goal_var.set("the goal")

    previous = FakeRunner(app._events)
    app._runner = previous

    app._start()

    assert factory.runners == [], "the next run started before the close"
    assert app._runner is previous, "the previous runner was replaced too early"

    previous.release_shutdown.set()
    _pump_until(app, lambda: bool(factory.runners))

    assert len(factory.runners) == 1
    url, goal, _settings, viewport, _grid_step = factory.runners[0].started
    assert url == "about:blank"
    assert goal == "the goal"
    assert viewport == (1280, 720)


@pytest.mark.gui
def test_start_does_not_block_the_tk_thread(monkeypatch, app):
    """The Tk thread returns from Start while the close is still running."""
    factory = RunnerFactory()
    monkeypatch.setattr(gui, "GuiRunner", factory)
    app._url_var.set("about:blank")
    app._goal_var.set("the goal")

    previous = FakeRunner(app._events)
    app._runner = previous

    started = time.monotonic()
    app._start()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, "the Tk thread waited for the close"
    assert factory.runners == [], "the next run started before the close"
    assert previous.shutdown_timeouts, "the previous runner was not shut down"
    previous.release_shutdown.set()


@pytest.mark.gui
def test_only_the_latest_start_survives_a_pending_close(monkeypatch, app):
    """A second Start replaces the run that waits for the close."""
    factory = RunnerFactory()
    monkeypatch.setattr(gui, "GuiRunner", factory)
    app._url_var.set("about:blank")
    app._goal_var.set("first goal")

    previous = FakeRunner(app._events)
    app._runner = previous

    app._start()
    app._goal_var.set("second goal")
    app._start()

    assert len(previous.shutdown_timeouts) == 1, "the close was started twice"

    previous.release_shutdown.set()
    _pump_until(app, lambda: bool(factory.runners))

    assert len(factory.runners) == 1
    assert factory.runners[0].started[1] == "second goal"
