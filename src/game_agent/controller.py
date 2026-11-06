"""Browser controller stub: the async boundary that drives the game page.

Concrete input/output arrives in Milestone 1. The controller owns a held
input ledger: the keys/buttons currently held down on the page.
"""

from dataclasses import dataclass

from game_agent.config import Settings
from game_agent.observation import Observation
from game_agent.schema import (
    Action,
    ClickAction,
    DoneAction,
    DragAction,
    HoldAction,
    MoveAction,
    PressAction,
    ReleaseAction,
    UncertainAction,
    WaitAction,
)


class ControllerError(Exception):
    """Base class for all browser-controller failures."""


class ActionNotAllowed(ControllerError):
    """The action violates the configured policy and must not be dispatched."""


class ActionExecutionError(ControllerError):
    """The action was allowed but failed while being performed on the page."""


@dataclass(frozen=True)
class BrowserOptions:
    """Browser launch options; viewport geometry lives in :class:`Settings`."""

    headless: bool = False
    user_data_dir: str = ".browser-profile"
    screenshot_dir: str = "screenshots"
    browser_channel: str | None = None


def _assert_coordinate_in_viewport(value: int, limit: int, name: str) -> None:
    """Raise unless coordinate *value* lies in the closed interval ``[0, limit]``."""
    if not 0 <= value <= limit:
        raise ActionNotAllowed(
            f"coordinate {name}={value} is outside the viewport (0..{limit})"
        )


def assert_executable(action: Action, settings: Settings) -> None:
    """Raise :class:`ActionNotAllowed` unless *action* may run under *settings*.

    Pure policy gate with no I/O, applied before dispatch: terminal reports
    are never executable; pointer coordinates must land inside the configured
    viewport; keys, buttons and durations must respect the configured
    allowlists and caps.
    """
    if isinstance(action, DoneAction | UncertainAction):
        raise ActionNotAllowed(
            f"{action.type!r} is a report, not an executable action"
        )

    if isinstance(action, ClickAction | MoveAction):
        _assert_coordinate_in_viewport(action.x, settings.viewport_width, "x")
        _assert_coordinate_in_viewport(action.y, settings.viewport_height, "y")
    elif isinstance(action, DragAction):
        _assert_coordinate_in_viewport(action.start_x, settings.viewport_width, "start_x")
        _assert_coordinate_in_viewport(action.start_y, settings.viewport_height, "start_y")
        _assert_coordinate_in_viewport(action.end_x, settings.viewport_width, "end_x")
        _assert_coordinate_in_viewport(action.end_y, settings.viewport_height, "end_y")

    if isinstance(action, PressAction | HoldAction | ReleaseAction):
        if action.key is not None and action.key not in settings.allowed_keys:
            raise ActionNotAllowed(f"key {action.key!r} is not in the allowed keys")

    if isinstance(action, ClickAction | DragAction | HoldAction | ReleaseAction):
        button = action.button
        if button is not None and button.value not in settings.allowed_buttons:
            raise ActionNotAllowed(
                f"button {button.value!r} is not in the allowed buttons"
            )

    if isinstance(action, WaitAction) and action.duration_ms > settings.max_wait_ms:
        raise ActionNotAllowed(
            f"wait duration_ms={action.duration_ms} exceeds cap {settings.max_wait_ms}"
        )
    if isinstance(action, HoldAction) and action.duration_ms > settings.max_hold_ms:
        raise ActionNotAllowed(
            f"hold duration_ms={action.duration_ms} exceeds cap {settings.max_hold_ms}"
        )
    if isinstance(action, DragAction) and action.duration_ms > settings.max_wait_ms:
        raise ActionNotAllowed(
            f"drag duration_ms={action.duration_ms} exceeds cap {settings.max_wait_ms}"
        )


class BrowserController:
    """Async facade over the game page (Milestone 1).

    Tracks held keys/buttons internally: every hold leaves an entry in a
    private ledger so ``release_all`` can always unwind them.
    """

    async def open(self) -> None:
        """Launch the browser, load the game page, and get it playable.

        Raises:
            NotImplementedError: Until Milestone 1.
        """
        raise NotImplementedError("Milestone 1: BrowserController.open")

    async def screenshot(self) -> Observation:
        """Capture the viewport as an :class:`Observation`.

        Raises:
            NotImplementedError: Until Milestone 1.
        """
        raise NotImplementedError("Milestone 1: BrowserController.screenshot")

    async def execute(self, action: Action) -> None:
        """Dispatch a schema-valid :class:`Action` to the game page.

        Args:
            action: The validated action to perform.

        Raises:
            NotImplementedError: Until Milestone 1.
        """
        raise NotImplementedError("Milestone 1: BrowserController.execute")

    async def release_all(self) -> None:
        """Release every held key/button and empty the held-input ledger.

        Safe-recovery primitive the agent loop calls from ``RECOVERING``:
        a real implementation tracks what is held and releases each key or
        button so recovery never leaves inputs stuck down.

        Raises:
            NotImplementedError: Until Milestone 1.
        """
        raise NotImplementedError("Milestone 1: BrowserController.release_all")

    async def close(self) -> None:
        """Shut the browser down cleanly.

        Raises:
            NotImplementedError: Until Milestone 1.
        """
        raise NotImplementedError("Milestone 1: BrowserController.close")
