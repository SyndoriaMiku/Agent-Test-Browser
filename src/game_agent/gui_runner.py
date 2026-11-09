"""Background runner for the desktop GUI: one agent session on its own
thread, publishing every event on a queue for the view.

The Tk thread must never block, so a run happens on a worker thread with
its own event loop (:meth:`GuiRunner.start`).  Everything the view needs to
know travels through one queue: the session's JSONL records
(``observation``/``decision``/``action``/``verify``/``model``) are published
by :class:`_QueueLogger`, and the runner adds the ``status``,
``session_end`` and ``error`` records of its own.

Browser lifecycle: the runner opens the browser before the session starts
and leaves it **open** when the run ends, so the user can look at the final
screen.  A stop request closes it instead -- the session stops at its next
cycle boundary (an action or a model call in flight is never interrupted)
and the browser is closed once ``run`` has returned.  :meth:`GuiRunner.shutdown`
does the same for a run that already ended: it stops the run, joins the
worker and closes the browser, so the view can close the browser of the
previous run before it starts a new one.

Whatever happens, the runner ends up closed: the close is attempted exactly
once and is marked as done even when there was no browser to close, so
:meth:`GuiRunner.join` always returns.

The worker keeps its event loop alive after the session ended: Playwright's
transport task lives on that loop, so closing it would leave the browser
open but uncontrollable -- and impossible to close later.  The loop is
closed together with the browser.
"""

import asyncio
import contextlib
import inspect
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Protocol

from game_agent.config import Settings
from game_agent.controller import BrowserController, BrowserOptions
from game_agent.logging import SessionLogger
from game_agent.observation import Observation
from game_agent.schema import Action
from game_agent.session import Session, VLMProtocol
from game_agent.session_state import Clock
from game_agent.vlm import VLMClient, load_vlm_config


class RunnerController(Protocol):
    """The browser-controller surface the runner drives.

    The session only needs a part of it (see
    :class:`game_agent.session.ControllerProtocol`); the runner additionally
    opens the browser, navigates to the start URL and closes it again.
    """

    async def open(self) -> None: ...

    async def goto(self, url: str) -> None: ...

    async def screenshot(self) -> Observation: ...

    async def execute(self, action: Action) -> None: ...

    async def release_all(self) -> None: ...

    async def close(self) -> None: ...


ControllerFactory = Callable[..., RunnerController]
"""Builds the controller of a run.

Called as ``factory(url, settings, viewport)``; a factory written as
``factory(url, settings)`` is supported as well.
"""

VlmFactory = Callable[[Settings], VLMProtocol]
"""Builds the vision model of a run; called as ``factory(settings)``."""


class _QueueLogger(SessionLogger):
    """A session logger that publishes every record on a queue as well.

    The dict put on the queue is the very record that is written to the
    JSONL file, so the view sees exactly what the log contains.
    """

    def __init__(
        self,
        path: str | Path,
        queue: "Queue[dict[str, Any]]",
        clock: Clock | None = None,
    ) -> None:
        """Open the JSONL log at *path* and publish its records on *queue*."""
        super().__init__(path, clock)
        self._queue = queue

    def _write(self, record: dict[str, object]) -> None:
        """Write *record* to the log file and publish it on the queue."""
        super()._write(record)
        self._queue.put(dict(record))


class GuiRunner:
    """Runs one agent session on a background thread and publishes its events.

    Args:
        queue: The queue the view drains; every event is put on it.
        controller_factory: Builds the browser controller of a run.  It is
            called as ``factory(url, settings, viewport)``; a factory that
            takes only ``(url, settings)`` works as well.
        vlm_factory: Builds the vision model of a run; called as
            ``factory(settings)``.
        session_kwargs: Extra keyword arguments for :class:`Session`
            (``clock``, ``sleep``, the safety limits, ...) plus the
            optional ``logger_out`` path of the JSONL session log.

    A runner drives a single run: :meth:`start` may be called once.  The
    view calls :meth:`shutdown` to close the browser of a finished run.
    """

    def __init__(
        self,
        queue: "Queue[dict[str, Any]]",
        *,
        controller_factory: ControllerFactory | None = None,
        vlm_factory: VlmFactory | None = None,
        session_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Store the queue, the factories and the session keyword arguments."""
        self._queue = queue
        self._controller_factory = controller_factory or _default_controller_factory
        self._vlm_factory = vlm_factory or _default_vlm_factory
        kwargs = dict(session_kwargs or {})
        self._logger_out = kwargs.pop("logger_out", None)
        # ``grid_step`` belongs to ``start``; a copy in ``session_kwargs``
        # would collide with the keyword the runner passes to the session.
        kwargs.pop("grid_step", None)
        self._session_kwargs = kwargs
        self._thread: threading.Thread | None = None
        self._session: Session | None = None
        self._controller: RunnerController | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_requested = False
        self._started = False
        self._browser_closed = False
        self._close_started = False
        self._loop_closed = False
        self._close_lock = threading.Lock()
        self._closed = threading.Event()
        self._worker_done = threading.Event()

    # --- control -------------------------------------------------------------

    def start(
        self,
        url: str,
        goal: str,
        settings: Settings,
        viewport: tuple[int, int] | None = None,
        grid_step: int | None = 100,
    ) -> None:
        """Start a run on a background thread.

        Args:
            url: The page the agent drives.
            goal: The goal handed to the model.
            settings: The agent's configuration.
            viewport: Optional ``(width, height)`` in CSS pixels; it
                overrides the viewport of *settings* for the browser and
                for the coordinate checks.
            grid_step: Spacing of the coordinate grid drawn for the model;
                ``None`` disables the grid.

        Raises:
            RuntimeError: If the runner was already started.
        """
        if self._started:
            raise RuntimeError("this runner was already started")
        self._started = True
        self._stop_requested = False
        self._closed.clear()
        self._thread = threading.Thread(
            target=self._worker,
            args=(url, goal, settings, viewport, grid_step),
            name="game-agent-gui-runner",
            daemon=True,
        )
        self._thread.start()

    def pause(self) -> None:
        """Ask the running session to wait at its next cycle boundary.

        The session keeps its place and continues where it left off once
        :meth:`resume` is called.  Without a running session this does
        nothing.
        """
        session = self._session
        if session is None or not self.is_running:
            return
        session.pause()
        self._queue.put({"event": "status", "status": "paused"})

    def resume(self) -> None:
        """Let a paused session continue at its next cycle boundary.

        Without a running session this does nothing.
        """
        session = self._session
        if session is None or not self.is_running:
            return
        session.resume()
        self._queue.put({"event": "status", "status": "running"})

    def stop(self) -> None:
        """Ask the session to stop; the browser closes once the run returns.

        The stop takes effect at the session's next cycle boundary: an
        action or a model call in flight is not interrupted.  When the run
        has already ended, the browser that was left open is closed now.
        """
        self._stop_requested = True
        session = self._session
        if session is not None:
            session.stop()
        if self._thread is None or self._worker_done.is_set():
            # The worker is gone (or never ran): nobody else closes the
            # browser, so the close starts here.
            self._close_browser_later()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the run to finish.

        After a stop the browser is closed as well, and the wait covers
        that close: a returned ``join`` means the browser is gone.

        Args:
            timeout: How long to wait, in seconds; ``None`` waits forever.
        """
        thread = self._thread
        if thread is None:
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        thread.join(timeout)
        if self._stop_requested:
            # A stop closes the browser; wait for that close as well.
            self._closed.wait(self._remaining(deadline))

    def shutdown(self, timeout: float | None = None) -> None:
        """Stop the run, join the worker and close the browser.

        The call is idempotent: a runner that is already closed stays
        closed, and the browser is closed exactly once.  The view calls it
        before a new Start -- to close the browser of the previous run --
        and when the window closes.

        Args:
            timeout: How long to wait for the worker and for the close, in
                seconds; ``None`` waits forever.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        self.stop()
        self.join(self._remaining(deadline))
        # A run that ended normally leaves its browser open: close it now.
        self._close_browser_later()
        self._closed.wait(self._remaining(deadline))

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        """Return the seconds left until *deadline*."""
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    @property
    def is_running(self) -> bool:
        """Whether the background run is still in progress."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    # --- the worker ----------------------------------------------------------

    def _worker(
        self,
        url: str,
        goal: str,
        settings: Settings,
        viewport: tuple[int, int] | None,
        grid_step: int | None,
    ) -> None:
        """Run the session in this thread's own event loop.

        The loop is created by hand instead of by :func:`asyncio.run`:
        ``asyncio.run`` cancels every pending task when it returns, and
        Playwright's connection is such a task -- the browser would stay
        open but could not be driven or closed any more.  The loop is
        closed together with the browser instead.
        """
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_until_complete(
                self._run(url, goal, settings, viewport, grid_step)
            )
        finally:
            # The run is over: from here on nobody else closes the
            # browser, so a stop that arrives now is handed over to the
            # late close.
            self._worker_done.set()
            if loop is not None:
                if self._stop_requested:
                    # A stop that landed while the run was tearing down
                    # still closes the browser.
                    with contextlib.suppress(Exception):
                        loop.run_until_complete(self._close_browser())
                if self._closed.is_set():
                    # The browser is gone; nothing needs the loop any more.
                    self._close_loop_once(loop)

    async def _run(
        self,
        url: str,
        goal: str,
        settings: Settings,
        viewport: tuple[int, int] | None,
        grid_step: int | None,
    ) -> None:
        """Open the browser, run the session, and publish what happened."""
        effective = _settings_with_viewport(settings, viewport)
        logger: _QueueLogger | None = None
        try:
            controller = _call_factory(
                self._controller_factory, (url, effective, viewport)
            )
            self._controller = controller
            await controller.open()
            await controller.goto(url)
            vlm = self._vlm_factory(effective)
            logger = _QueueLogger(
                self._logger_out or _default_log_path(),
                self._queue,
                self._session_kwargs.get("clock"),
            )
            session = Session(
                controller,
                vlm,
                effective,
                logger=logger,
                grid_step=grid_step,
                **self._session_kwargs,
            )
            self._session = session
            if self._stop_requested:
                # The stop arrived before the session existed: apply it now,
                # so the session ends at its first cycle boundary.
                session.stop()
            self._queue.put({"event": "status", "status": "running"})
            result = await session.run(goal)
            self._queue.put(
                {
                    "event": "session_end",
                    "state": str(result.state),
                    "cycles": result.cycles,
                    "reason": result.reason,
                }
            )
            if not self._stop_requested:
                # The run ended on its own; the browser stays open for the
                # user to look at.
                self._queue.put({"event": "status", "status": "finished"})
        except Exception as exc:
            self._queue.put({"event": "error", "message": str(exc)})
        finally:
            if self._stop_requested:
                await self._close_browser()
            if logger is not None:
                with contextlib.suppress(Exception):
                    logger.close()

    # --- the browser ---------------------------------------------------------

    async def _close_browser(self) -> None:
        """Release the held inputs and close the browser, exactly once.

        The runner is marked as closed even when there is no browser to
        close -- a browser that never opened leaves nothing to wait for --
        so a waiting :meth:`join` is always released.
        """
        with self._close_lock:
            if self._browser_closed:
                return
            self._browser_closed = True
        controller = self._controller
        try:
            if controller is not None:
                with contextlib.suppress(Exception):
                    await controller.release_all()
                with contextlib.suppress(Exception):
                    await controller.close()
        finally:
            self._closed.set()
            self._queue.put({"event": "status", "status": "stopped"})

    def _close_browser_later(self) -> None:
        """Close the browser of a finished run on the runner's own loop.

        The browser belongs to the event loop of the worker thread, so the
        close runs there as well -- a browser that was left open by a
        finished run has to be closed from that loop to stay controllable.
        """
        if self._thread is None:
            # The runner never ran: there is no browser to close.
            self._closed.set()
            return
        if not self._worker_done.is_set():
            # The worker still owns the loop; it closes the browser itself.
            return
        loop = self._loop
        if loop is None:
            # The worker died before it could open a browser.
            self._closed.set()
            return
        with self._close_lock:
            if self._browser_closed:
                # The close is done or on its way; it marks the runner as
                # closed when it finishes.
                return
            if self._close_started:
                # A close is already on its way.
                return
            self._close_started = True
        threading.Thread(
            target=self._close_on_loop,
            args=(loop,),
            name="game-agent-gui-close",
            daemon=True,
        ).start()

    def _close_on_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Close the browser on *loop* and close the loop with it."""
        worker = self._thread
        if worker is not None and worker is not threading.current_thread():
            # The worker owns the loop until it returns.
            worker.join()
        asyncio.set_event_loop(loop)
        try:
            with contextlib.suppress(Exception):
                loop.run_until_complete(self._close_browser())
        finally:
            self._close_loop_once(loop)
            if not self._closed.is_set():
                # The close could not run at all: there is nothing left to
                # wait for, so the runner counts as closed.
                self._closed.set()

    def _close_loop_once(self, loop: asyncio.AbstractEventLoop) -> None:
        """Close *loop* once; a loop that is still in use is left open."""
        with self._close_lock:
            if self._loop_closed:
                return
            try:
                loop.close()
            except Exception:
                # The loop is still running: whoever owns it closes it.
                return
            self._loop_closed = True


# --- defaults and helpers ----------------------------------------------------


def _default_controller_factory(
    url: str, settings: Settings, viewport: tuple[int, int] | None = None
) -> BrowserController:
    """Build the real, headed browser controller of a run.

    The viewport lives in :class:`Settings` (the browser and the pointer
    coordinates share one CSS pixel space), so the factory only has to ask
    for a headed browser.
    """
    return BrowserController(settings, BrowserOptions(headless=False))


def _default_vlm_factory(settings: Settings) -> VLMClient:
    """Build the real vision model of a run from the environment."""
    return VLMClient(load_vlm_config())


def _default_log_path() -> Path:
    """Return the JSONL path of a run that names none."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("logs") / f"session-{stamp}.jsonl"


def _settings_with_viewport(
    settings: Settings, viewport: tuple[int, int] | None
) -> Settings:
    """Return *settings* with the viewport dimensions applied."""
    if viewport is None:
        return settings
    width, height = viewport
    return replace(settings, viewport_width=int(width), viewport_height=int(height))


def _call_factory(factory: Callable[..., Any], args: tuple[Any, ...]) -> Any:
    """Call *factory* with the leading arguments of *args* that it accepts.

    A controller factory may be written as ``(url, settings)`` or as
    ``(url, settings, viewport)``; both are called the way they are
    written.
    """
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return factory(*args)
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters.values()):
        return factory(*args)
    positional = [
        p
        for p in parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return factory(*args[: len(positional)])
