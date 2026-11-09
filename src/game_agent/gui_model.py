"""Pure display logic for the desktop GUI: event formatting, preview paths,
and token/cost accounting.

This is the GUI's model layer, kept free of Tk so it can be unit-tested
offline: :func:`format_event` renders the session event dicts (the JSONL
records written by the session logger plus the runner's own ``status``/
``session_end``/``error`` records) into compact human-readable strings,
:func:`preview_path` picks the screenshot the preview pane should show,
and :class:`UsageTracker` accumulates the model's token usage and prices
it (costs are USD; prices are per 1,000,000 tokens).  The only I/O in
the module is the file-existence check inside :func:`preview_path`.

Rendering is defensive by design: every field is read with ``.get`` and
unknown or malformed records fall back to a generic string instead of
raising, so a broken event can never take the GUI down.

One wording matters beyond cosmetics: an ``action`` record's ``accepted``
flag means the *validator* accepted the decision, not that the action
executed successfully (the record is written even when ``execute``
raises).  It is therefore rendered as ``accepted (validation)``.
"""

from pathlib import Path

_MILLION = 1_000_000
"""Token count the per-million prices of :meth:`UsageTracker.cost_usd` refer to."""


# --- event formatting --------------------------------------------------------


def format_event(event: dict) -> str:
    """Render one event dict as a compact, human-readable string.

    Known kinds get a dedicated rendering: the session logger's
    ``observation``/``decision``/``action``/``verify``/``model`` records
    and the GUI runner's ``status``/``session_end``/``error`` records.
    Anything else -- an unknown kind, a non-dict, missing fields -- falls
    back to a generic string; this never raises.
    """
    kind = event.get("event") if isinstance(event, dict) else None
    renderer = _RENDERERS.get(kind) if isinstance(kind, str) else None
    if renderer is None:
        return str(event)
    return renderer(event)


def _format_observation(event: dict) -> str:
    """``observation`` record: which screen was captured."""
    path = event.get("screenshot_path")
    shot = Path(path).name if path else "no screenshot"
    return f"observation {event.get('observation_id')}  screenshot: {shot}"


def _format_decision(event: dict) -> str:
    """``decision`` record: state, chosen action, confidence, reason."""
    decision = event.get("decision")
    if not isinstance(decision, dict):
        decision = {}
    head = (
        f"decision {event.get('observation_id')}"
        f"  [{decision.get('state') or '?'}] {_format_action(decision.get('action'))}"
    )
    confidence = decision.get("confidence")
    if confidence is not None:
        head += f"  confidence {confidence}"
    reason = decision.get("reason")
    if reason:
        head += f"\n  reason: {reason}"
    return head


def _format_action_record(event: dict) -> str:
    """``action`` record: what was (attempted to be) executed.

    ``accepted`` is the *validator's* verdict -- the record is also
    written when execution itself fails -- so it reads ``accepted
    (validation)``, never "success".
    """
    line = (
        f"action {event.get('observation_id')}  {_format_action(event.get('action'))}"
    )
    if event.get("accepted") is not None:
        if event["accepted"]:
            line += "  accepted (validation)"
        else:
            line += "  rejected"
            if event.get("reason"):
                line += f" ({event['reason']})"
    return line


def _format_verify(event: dict) -> str:
    """``verify`` record: whether the screen changed after the action."""
    outcome = "screen changed" if event.get("changed") else "screen unchanged"
    line = f"verify {event.get('observation_id')}  {outcome}"
    note = event.get("note")
    # The session echoes the outcome as the note by default; skip that
    # exact echo so the line does not read "screen changed — screen changed".
    if note and note != outcome:
        line += f"  — {note}"
    return line


def _format_model(event: dict) -> str:
    """``model`` record: model name and the tokens it charged."""
    return (
        f"model {event.get('observation_id')}  {event.get('model') or '?'}  tokens:"
        f" {event.get('prompt_tokens') or 0} in /"
        f" {event.get('completion_tokens') or 0} out /"
        f" {event.get('total_tokens') or 0} total"
    )


def _format_status(event: dict) -> str:
    """Runner ``status`` record."""
    line = f"status {event.get('status')}"
    if event.get("detail"):
        line += f"  — {event['detail']}"
    return line


def _format_session_end(event: dict) -> str:
    """Runner ``session_end`` record: how the session finished."""
    line = (
        f"session_end  state={event.get('state')}  cycles={event.get('cycles')}"
    )
    if event.get("reason"):
        line += f"  reason: {event['reason']}"
    return line


def _format_error(event: dict) -> str:
    """Runner ``error`` record."""
    return f"error  {event.get('message')}"


_RENDERERS = {
    "observation": _format_observation,
    "decision": _format_decision,
    "action": _format_action_record,
    "verify": _format_verify,
    "model": _format_model,
    "status": _format_status,
    "session_end": _format_session_end,
    "error": _format_error,
}


def _format_action(action: object) -> str:
    """Render one action dict compactly, e.g. ``click @ (12, 34)``."""
    if not isinstance(action, dict):
        return "(no action)" if action is None else str(action)
    kind = action.get("type")
    if kind == "click":
        line = f"click @ ({action.get('x')}, {action.get('y')})"
        if action.get("button"):
            line += f" {action['button']}"
        if action.get("click_count"):
            line += f" x{action['click_count']}"
        return line
    if kind == "move":
        return f"move @ ({action.get('x')}, {action.get('y')})"
    if kind == "press":
        return f"press {action.get('key')!r}"
    if kind == "hold":
        return f"hold {_target(action)} for {action.get('duration_ms')}ms"
    if kind == "release":
        return f"release {_target(action)}"
    if kind == "drag":
        return (
            f"drag ({action.get('start_x')}, {action.get('start_y')})"
            f" -> ({action.get('end_x')}, {action.get('end_y')})"
        )
    if kind == "wheel":
        return f"wheel (dx={action.get('delta_x', 0)}, dy={action.get('delta_y', 0)})"
    if kind == "wait":
        return f"wait {action.get('duration_ms')}ms"
    if kind == "done":
        return f"done: {action.get('summary')}"
    if kind == "uncertain":
        return f"uncertain: {action.get('reason')}"
    return str(action)


def _target(action: dict) -> str:
    """The key or button a ``hold``/``release`` action aims at."""
    key = action.get("key")
    if key:
        return f"key {key!r}"
    return f"button {action.get('button')!r}"


# --- preview path ------------------------------------------------------------


def preview_path(screenshot_path: str | None) -> str | None:
    """Pick the screenshot the preview pane should show.

    The grid image (``<stem>.grid.png`` next to the capture) is preferred
    because its coordinate labels match what the model saw; without one --
    ``grid_step=None`` in the session, or the grid not written yet -- the
    original path is returned as-is (even when the file does not exist
    yet, so the view keeps working with whatever it can load).  ``None``
    (or an empty path) yields ``None``.
    """
    if not screenshot_path:
        return None
    raw = Path(screenshot_path)
    grid = raw.with_name(f"{raw.stem}.grid.png")
    return str(grid) if grid.exists() else str(raw)


# --- token/cost accounting ----------------------------------------------------


class UsageTracker:
    """Accumulates the token counters reported by the model.

    ``add`` sums one ``usage`` dict (missing keys count as zero), the
    three properties expose the running totals, and :meth:`cost_usd`
    prices them when both per-million prices are known.
    """

    def __init__(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0

    def add(self, usage: dict | None) -> None:
        """Add one usage report; missing keys (or ``None``) add nothing."""
        usage = usage or {}
        self._prompt_tokens += usage.get("prompt_tokens") or 0
        self._completion_tokens += usage.get("completion_tokens") or 0
        self._total_tokens += usage.get("total_tokens") or 0

    @property
    def prompt_tokens(self) -> int:
        """Total prompt tokens reported so far."""
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        """Total completion tokens reported so far."""
        return self._completion_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens (as reported) so far."""
        return self._total_tokens

    def cost_usd(
        self, in_per_1m: float | None, out_per_1m: float | None
    ) -> float | None:
        """Return the accumulated cost in USD, or ``None`` without both prices.

        ``in_per_1m`` / ``out_per_1m`` are prices in **USD per 1,000,000
        tokens** for prompt / completion tokens respectively (so ``1.0``
        means one dollar per million prompt tokens).  The cost normalizes
        the token totals to millions and applies the prices directly:
        ``prompt / 1e6 * in_per_1m + completion / 1e6 * out_per_1m``.
        """
        if in_per_1m is None or out_per_1m is None:
            return None
        prompt_cost = self._prompt_tokens / _MILLION * in_per_1m
        completion_cost = self._completion_tokens / _MILLION * out_per_1m
        return prompt_cost + completion_cost
