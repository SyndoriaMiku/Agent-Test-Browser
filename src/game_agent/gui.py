"""Tkinter desktop view for observing and steering the browser agent.

The view owns no agent logic.  A :class:`~game_agent.gui_runner.GuiRunner`
runs the session on a worker thread and publishes events on a queue; this
module periodically drains that queue, appends human-readable lines to the
log, updates the screenshot preview, and refreshes the status bar.

Run the application with ``python -m game_agent.gui``.
"""

from __future__ import annotations

import math
import os
import queue
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk
from typing import Any

from PIL import Image, ImageTk

from game_agent.config import Settings, load_settings
from game_agent.gui_model import UsageTracker, format_event, preview_path
from game_agent.gui_runner import GuiRunner

_POLL_INTERVAL_MS = 150
_PREVIEW_MAX_SIZE = (720, 480)


def _default_url() -> str:
    """Return the demo page as a ``file://`` URI."""
    return Path("examples/demo.html").resolve().as_uri()


def _read_prices() -> tuple[float | None, float | None]:
    """Read optional USD-per-million token prices from the environment.

    Cost is shown only when both prices are present and valid.  The model
    API key is deliberately never read by this module.
    """
    input_raw = os.environ.get("NVIDIA_PRICE_INPUT_PER_1M")
    output_raw = os.environ.get("NVIDIA_PRICE_OUTPUT_PER_1M")
    if input_raw is None or output_raw is None:
        return None, None
    try:
        input_price = float(input_raw)
        output_price = float(output_raw)
    except ValueError:
        return None, None
    if not math.isfinite(input_price) or not math.isfinite(output_price):
        return None, None
    if input_price < 0 or output_price < 0:
        return None, None
    return input_price, output_price


class GameAgentGui(tk.Tk):
    """Tk window for starting and observing an agent session."""

    def __init__(self) -> None:
        """Build the window and begin polling the runner queue."""
        super().__init__()
        self.title("Game Agent")
        self.geometry("1200x760")
        self.minsize(900, 600)

        self._url_var = tk.StringVar(value=_default_url())
        self._goal_var = tk.StringVar()
        self._width_var = tk.StringVar(value="1280")
        self._height_var = tk.StringVar(value="720")
        self._status_var = tk.StringVar(value="State: IDLE")

        self._runner: GuiRunner | None = None
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._usage = UsageTracker()
        self._input_price, self._output_price = _read_prices()
        self._runner_status = "idle"
        self._state_name = "IDLE"
        self._cycles = 0
        self._remaining: int | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._closed = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(_POLL_INTERVAL_MS, self._poll)

    def _build_ui(self) -> None:
        """Create the controls, preview, log, and status bar."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        controls = ttk.Frame(self, padding=(8, 8, 8, 4))
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=3)
        controls.columnconfigure(3, weight=3)

        ttk.Label(controls, text="URL").grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(controls, textvariable=self._url_var, width=36).grid(
            row=0, column=1, sticky="ew", padx=(0, 8)
        )
        ttk.Label(controls, text="Goal").grid(row=0, column=2, padx=(0, 4))
        ttk.Entry(controls, textvariable=self._goal_var, width=36).grid(
            row=0, column=3, sticky="ew", padx=(0, 8)
        )
        ttk.Label(controls, text="W").grid(row=0, column=4, padx=(0, 4))
        ttk.Entry(controls, textvariable=self._width_var, width=6).grid(
            row=0, column=5, padx=(0, 8)
        )
        ttk.Label(controls, text="H").grid(row=0, column=6, padx=(0, 4))
        ttk.Entry(controls, textvariable=self._height_var, width=6).grid(
            row=0, column=7, padx=(0, 8)
        )

        buttons = ttk.Frame(controls)
        buttons.grid(row=0, column=8, sticky="e")
        self._start_button = ttk.Button(buttons, text="Start", command=self._start)
        self._pause_button = ttk.Button(buttons, text="Pause", command=self._pause)
        self._resume_button = ttk.Button(buttons, text="Resume", command=self._resume)
        self._stop_button = ttk.Button(buttons, text="Stop", command=self._stop)
        for button in (
            self._start_button,
            self._pause_button,
            self._resume_button,
            self._stop_button,
        ):
            button.pack(side="left", padx=(0, 4))

        content = ttk.Frame(self, padding=(8, 4, 8, 4))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(content, text="Preview", padding=4)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self._preview_label = ttk.Label(
            preview_frame,
            text="No preview yet",
            anchor="center",
        )
        self._preview_label.grid(row=0, column=0, sticky="nsew")

        log_frame = ttk.LabelFrame(content, text="Log", padding=4)
        log_frame.grid(row=0, column=1, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self._log = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            width=48,
            height=24,
            state="disabled",
        )
        self._log.grid(row=0, column=0, sticky="nsew")

        status = ttk.Label(
            self,
            textvariable=self._status_var,
            anchor="w",
            padding=(8, 5),
            relief="sunken",
        )
        status.grid(row=2, column=0, sticky="ew")

        self._update_buttons()

    def _start(self) -> None:
        """Stop any previous run, then start a fresh session."""
        url = self._url_var.get().strip()
        goal = self._goal_var.get().strip()
        viewport = self._read_viewport()
        if not url:
            self._report_error("URL is required")
            return
        if not goal:
            self._report_error("Goal is required")
            return
        if viewport is None:
            return

        if self._runner is not None:
            self._status_var.set("State: STOPPING")
            self.update_idletasks()
            try:
                self._runner.shutdown()
            except Exception as exc:
                self._report_error(f"Could not stop the previous run: {exc}")
                return

        try:
            settings: Settings = load_settings()
        except (TypeError, ValueError) as exc:
            self._report_error(f"Invalid settings: {exc}")
            return

        self._events = queue.Queue()
        self._usage = UsageTracker()
        self._runner_status = "starting"
        self._state_name = "IDLE"
        self._cycles = 0
        self._remaining = None
        self._preview_photo = None
        self._preview_label.configure(image="", text="No preview yet")
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")
        self._refresh_status()

        runner = GuiRunner(self._events)
        try:
            runner.start(
                url,
                goal,
                settings,
                viewport=viewport,
                grid_step=100,
            )
        except Exception as exc:
            self._runner = None
            self._report_error(f"Could not start the run: {exc}")
            return
        self._runner = runner
        self._refresh_status()

    def _pause(self) -> None:
        """Ask the current run to pause at its next cycle boundary."""
        if self._runner is not None:
            self._runner.pause()

    def _resume(self) -> None:
        """Resume the current paused run."""
        if self._runner is not None:
            self._runner.resume()

    def _stop(self) -> None:
        """Ask the current run to stop and close its browser."""
        if self._runner is not None:
            self._runner_status = "stopping"
            self._runner.stop()
            self._update_buttons()

    def _poll(self) -> None:
        """Drain all queued runner events and refresh the view."""
        if self._closed:
            return
        try:
            while True:
                try:
                    event = self._events.get_nowait()
                except queue.Empty:
                    break
                self._handle_event(event)
        finally:
            self._refresh_status()
            self._update_buttons()
            if not self._closed:
                self.after(_POLL_INTERVAL_MS, self._poll)

    def _handle_event(self, event: dict[str, Any]) -> None:
        """Apply one runner event to the log, preview, and status."""
        self._append_log(format_event(event))
        kind = event.get("event") if isinstance(event, dict) else None

        if kind == "observation":
            path = preview_path(event.get("screenshot_path"))
            if path:
                self._show_preview(path)
        elif kind == "model":
            usage = event.get("usage")
            if not isinstance(usage, dict):
                usage = {
                    "prompt_tokens": event.get("prompt_tokens"),
                    "completion_tokens": event.get("completion_tokens"),
                    "total_tokens": event.get("total_tokens"),
                }
            self._usage.add(usage)
            self._refresh_status()
        elif kind == "status":
            status = event.get("status")
            if isinstance(status, str):
                self._runner_status = status
                if status not in {"running", "paused"}:
                    self._state_name = status.upper()
        elif kind == "session_end":
            state = event.get("state")
            if isinstance(state, str):
                self._state_name = state.removeprefix("State.").upper()
            cycles = event.get("cycles")
            if isinstance(cycles, int):
                self._cycles = cycles
            self._runner_status = "finished"
        elif kind == "error":
            self._runner_status = "error"
            self._state_name = "ERROR"

    def _show_preview(self, path: str) -> None:
        """Load and display a screenshot, scaled to keep its aspect ratio."""
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail(_PREVIEW_MAX_SIZE, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image, master=self)
        except (OSError, ValueError, tk.TclError) as exc:
            self._append_log(f"preview unavailable: {exc}")
            return
        self._preview_photo = photo
        self._preview_label.configure(image=photo, text="")

    def _append_log(self, line: str) -> None:
        """Append one line to the read-only log."""
        self._log.configure(state="normal")
        self._log.insert("end", line + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _report_error(self, message: str) -> None:
        """Show an error in the log and status bar without raising."""
        self._append_log(format_event({"event": "error", "message": message}))
        self._state_name = "ERROR"
        self._refresh_status()

    def _read_viewport(self) -> tuple[int, int] | None:
        """Read and validate the viewport dimensions from the entries."""
        try:
            width = int(self._width_var.get())
            height = int(self._height_var.get())
        except ValueError:
            self._report_error("Viewport width and height must be integers")
            return None
        if width <= 0 or height <= 0:
            self._report_error("Viewport width and height must be positive")
            return None
        return width, height

    def _refresh_status(self) -> None:
        """Refresh the status line from the session and usage tracker."""
        session = getattr(self._runner, "_session", None) if self._runner else None
        if session is not None:
            state = getattr(session, "state", None)
            state_name = getattr(state, "name", str(state))
            self._state_name = state_name.upper()
            cycles = getattr(session, "cycles", self._cycles)
            remaining = getattr(session, "remaining_actions", self._remaining)
            if isinstance(cycles, int):
                self._cycles = cycles
            if isinstance(remaining, int):
                self._remaining = remaining

        if self._runner_status == "paused":
            state = f"PAUSED ({self._state_name})"
        else:
            state = self._state_name

        remaining_text = "—" if self._remaining is None else str(self._remaining)
        tokens = (
            f"{self._usage.total_tokens:,} "
            f"({self._usage.prompt_tokens:,} in / "
            f"{self._usage.completion_tokens:,} out)"
        )
        parts = [
            f"State: {state}",
            f"Cycles: {self._cycles}",
            f"Remaining: {remaining_text}",
            f"Tokens: {tokens}",
        ]
        cost = self._usage.cost_usd(self._input_price, self._output_price)
        if cost is not None:
            parts.append(f"Cost: ${cost:.6f}")
        self._status_var.set(" | ".join(parts))

    def _update_buttons(self) -> None:
        """Enable controls that can act on the runner's current state."""
        running = self._runner is not None and self._runner.is_running
        self._pause_button.configure(
            state="normal" if running and self._runner_status != "paused" else "disabled"
        )
        self._resume_button.configure(
            state="normal" if running and self._runner_status == "paused" else "disabled"
        )
        self._stop_button.configure(
            state="normal" if self._runner is not None else "disabled"
        )

    def _close(self) -> None:
        """Shut down the runner and close the window."""
        self._closed = True
        try:
            if self._runner is not None:
                self._runner.shutdown()
        finally:
            self.destroy()


def main() -> None:
    """Open the desktop GUI."""
    app = GameAgentGui()
    app.mainloop()


if __name__ == "__main__":
    main()
