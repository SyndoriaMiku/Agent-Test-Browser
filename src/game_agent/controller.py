"""Browser controller: the async boundary that drives the game page.

The controller owns the Playwright lifecycle (one persistent browser context
per controller) and a held input ledger: the keys/buttons currently held down
on the page.
"""

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import (
    BrowserContext,
    Error as PlaywrightError,
    Page,
    Playwright,
    async_playwright,
)

from game_agent.config import Settings
from game_agent.keys import to_playwright_key
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
    WheelAction,
)
from game_agent.session_state import Clock, MonotonicClock

_NEXT_FRAME_SCRIPT = (
    "() => new Promise(resolve => requestAnimationFrame(() => resolve(true)))"
)

# A drag interpolates its path with one mouse move per this many
# milliseconds of the requested duration, capped so a long drag cannot emit
# an unbounded number of events.
_DRAG_STEP_MS = 10
_MAX_DRAG_STEPS = 100


def _drag_steps(duration_ms: int) -> int:
    """Return how many interpolated mouse moves a drag of *duration_ms* uses."""
    return max(1, min(_MAX_DRAG_STEPS, round(duration_ms / _DRAG_STEP_MS)))


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

    Owns one persistent Playwright browser context sized from ``Settings``:
    the screenshot and the pointer coordinates therefore share the same CSS
    pixel space.  Tracks the pointer position and the held keys/buttons
    internally: every hold leaves an entry in a private ledger so
    ``release_all`` can always unwind them.
    """

    def __init__(
        self,
        settings: Settings,
        options: BrowserOptions | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Store the configuration; nothing is launched until :meth:`open`.

        Args:
            settings: Viewport geometry and policy limits — the single source
                of truth for the browser's size.
            options: Browser launch options; defaults to :class:`BrowserOptions`.
            clock: Time source for observation timestamps; defaults to a
                monotonic clock.
        """
        self._settings = settings
        self._options = BrowserOptions() if options is None else options
        self._clock: Clock = MonotonicClock() if clock is None else clock
        self._session_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self._observation_count = 0
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._held_keys: set[str] = set()
        self._held_buttons: set[str] = set()
        # The pointer starts at the centre of the viewport so a wheel action
        # never scrolls at (0, 0).
        self._cursor: tuple[int, int] = (
            self._settings.viewport_width // 2,
            self._settings.viewport_height // 2,
        )

    @property
    def held_keys(self) -> frozenset[str]:
        """The keys currently held down, as a read-only snapshot."""
        return frozenset(self._held_keys)

    @property
    def held_buttons(self) -> frozenset[str]:
        """The mouse buttons currently held down, as a read-only snapshot."""
        return frozenset(self._held_buttons)

    @property
    def page(self) -> Page:
        """The Playwright page the controller drives.

        Raises:
            ControllerError: If the browser has not been opened yet.
        """
        if self._page is None:
            raise ControllerError("browser is not open; call open() first")
        return self._page

    async def open(self) -> None:
        """Launch the browser and hand back a page ready to drive.

        Starts Playwright and launches a persistent Chromium context in the
        configured user data directory, sized from ``settings`` (viewport and
        device scale factor) and ``options`` (headless flag, browser channel).
        The context is not navigated here: use :meth:`goto` for that.

        Raises:
            ControllerError: If a browser is already open.
            playwright.async_api.Error: If the browser cannot be launched.
        """
        if self._context is not None:
            raise ControllerError("browser is already open")

        playwright = await async_playwright().start()
        try:
            context = await playwright.chromium.launch_persistent_context(
                self._options.user_data_dir,
                headless=self._options.headless,
                viewport={
                    "width": self._settings.viewport_width,
                    "height": self._settings.viewport_height,
                },
                device_scale_factor=self._settings.device_scale_factor,
                channel=self._options.browser_channel,
            )
        except BaseException:
            with contextlib.suppress(Exception):
                await playwright.stop()
            raise

        pages = context.pages
        page = pages[0] if pages else await context.new_page()
        self._playwright = playwright
        self._context = context
        self._page = page

    async def goto(self, url: str) -> None:
        """Navigate the current page to *url* and wait for the load event.

        Args:
            url: The URL to open (for example a ``file://`` URI).

        Raises:
            ControllerError: If the browser has not been opened yet.
        """
        await self.page.goto(url, wait_until="load")

    async def screenshot(self) -> Observation:
        """Capture the viewport as an :class:`Observation`.

        The PNG is written to ``<screenshot_dir>/<session_id>/<observation_id>.png``;
        the viewport numbers come from ``settings`` and the timestamp from the
        injected clock.

        Raises:
            ControllerError: If the browser has not been opened yet.
        """
        page = self.page
        observation_id = self._next_observation_id()
        session_dir = Path(self._options.screenshot_dir) / self._session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = session_dir / f"{observation_id}.png"
        await self._capture_viewport(page, screenshot_path)
        return Observation(
            observation_id=observation_id,
            screenshot_path=str(screenshot_path),
            viewport_width=self._settings.viewport_width,
            viewport_height=self._settings.viewport_height,
            device_scale_factor=self._settings.device_scale_factor,
            timestamp=self._clock.now(),
        )

    async def execute(self, action: Action) -> None:
        """Dispatch a schema-valid :class:`Action` to the game page.

        The action is first checked against the controller's own policy
        (:func:`assert_executable`) and only then performed.  ``hold`` and
        ``drag`` are atomic: once the input is down it is released in a
        ``finally`` block, so a failure during the hold never leaves it
        stuck; the ledger keeps it tracked until a release actually
        succeeded, so ``release_all`` can retry it later.

        Args:
            action: The validated action to perform.

        Raises:
            ActionNotAllowed: If the action violates the configured policy
                (terminal reports included).
            ControllerError: If the browser is not open yet.
        """
        assert_executable(action, self._settings)
        page = self.page

        if isinstance(action, ClickAction):
            await page.mouse.click(
                action.x,
                action.y,
                button=action.button.value,
                click_count=action.click_count,
            )
            self._cursor = (action.x, action.y)
        elif isinstance(action, MoveAction):
            await page.mouse.move(action.x, action.y)
            self._cursor = (action.x, action.y)
        elif isinstance(action, PressAction):
            await page.keyboard.press(to_playwright_key(action.key))
        elif isinstance(action, HoldAction):
            await self._hold(action)
        elif isinstance(action, ReleaseAction):
            await self._release(action)
        elif isinstance(action, DragAction):
            await self._drag(action)
        elif isinstance(action, WheelAction):
            await page.mouse.move(*self._cursor)
            await page.mouse.wheel(action.delta_x, action.delta_y)
        elif isinstance(action, WaitAction):
            await asyncio.sleep(action.duration_ms / 1000)
        else:  # pragma: no cover - assert_executable rejects reports first
            raise ActionNotAllowed(f"action type {action.type!r} is not executable")

    async def _hold(self, action: HoldAction) -> None:
        """Hold the action's key or button down, then release it again."""
        if action.key is not None:
            await self._hold_key(action.key)
            try:
                await asyncio.sleep(action.duration_ms / 1000)
            finally:
                await self._release_key(action.key)
            return

        button = action.button
        if button is None:  # pragma: no cover - the schema demands one target
            raise ActionNotAllowed("hold action names neither a key nor a button")
        await self._hold_button(button.value)
        try:
            await asyncio.sleep(action.duration_ms / 1000)
        finally:
            await self._release_button(button.value)

    async def _release(self, action: ReleaseAction) -> None:
        """Release the action's key or button; a no-op when it is not held."""
        if action.key is not None:
            if action.key in self._held_keys:
                await self._release_key(action.key)
            return

        button = action.button
        if button is None:  # pragma: no cover - the schema demands one target
            raise ActionNotAllowed("release action names neither a key nor a button")
        if button.value in self._held_buttons:
            await self._release_button(button.value)

    async def _drag(self, action: DragAction) -> None:
        """Drag from the start point to the end point along an interpolated path."""
        page = self.page
        await page.mouse.move(action.start_x, action.start_y)
        self._cursor = (action.start_x, action.start_y)
        await self._hold_button(action.button.value)
        try:
            await page.mouse.move(
                action.end_x,
                action.end_y,
                steps=_drag_steps(action.duration_ms),
            )
            self._cursor = (action.end_x, action.end_y)
        finally:
            await self._release_button(action.button.value)

    async def _hold_key(self, key: str) -> None:
        """Press *key* down and track it in the held-key ledger.

        The key enters the ledger *before* the browser is asked to press it,
        so even a failed press leaves an entry for ``release_all`` to clean
        up.
        """
        self._held_keys.add(key)
        await self.page.keyboard.down(to_playwright_key(key))

    async def _release_key(self, key: str) -> None:
        """Release *key* and drop it from the ledger once the browser is up."""
        await self.page.keyboard.up(to_playwright_key(key))
        self._held_keys.discard(key)

    async def _hold_button(self, button: str) -> None:
        """Press mouse *button* down and track it in the held-button ledger."""
        self._held_buttons.add(button)
        await self.page.mouse.down(button=button)

    async def _release_button(self, button: str) -> None:
        """Release mouse *button* and drop it once the browser is up."""
        await self.page.mouse.up(button=button)
        self._held_buttons.discard(button)

    async def release_all(self) -> None:
        """Release every held key/button and empty the held-input ledger.

        Safe-recovery primitive the agent loop calls from ``RECOVERING``:
        it releases each key or button the ledger still lists so recovery
        never leaves inputs stuck down.  Releases are best-effort: a failing
        release is swallowed so one bad input cannot block the others, and
        its ledger entry stays for the next attempt.
        """
        if self._page is None:
            # Nothing can be held down in a browser that is not running.
            self._held_keys.clear()
            self._held_buttons.clear()
            return

        for key in list(self._held_keys):
            with contextlib.suppress(Exception):
                await self._release_key(key)
        for button in list(self._held_buttons):
            with contextlib.suppress(Exception):
                await self._release_button(button)

    async def close(self) -> None:
        """Shut the browser down cleanly.

        Releases held inputs first, then closes the browser context and stops
        Playwright; the browser is closed even when releasing fails.
        """
        try:
            await self.release_all()
        finally:
            await self._shutdown_browser()

    def _next_observation_id(self) -> str:
        """Return a fresh observation id, unique within the session."""
        self._observation_count += 1
        return f"obs-{self._observation_count:04d}-{uuid.uuid4().hex[:6]}"

    async def _capture_viewport(self, page: Page, path: Path) -> None:
        """Capture the viewport into *path*, retrying once on failure.

        Chromium occasionally answers a capture request before the page has a
        frame to hand over ("Unable to capture screenshot"); waiting for the
        next animation frame and asking again recovers from that.
        """
        try:
            await page.screenshot(path=str(path))
        except PlaywrightError:
            await page.evaluate(_NEXT_FRAME_SCRIPT)
            await page.screenshot(path=str(path))

    async def _shutdown_browser(self) -> None:
        """Close the browser context and stop Playwright, best-effort."""
        context, self._context = self._context, None
        self._page = None
        playwright, self._playwright = self._playwright, None
        try:
            if context is not None:
                with contextlib.suppress(PlaywrightError):
                    await context.close()
        finally:
            if playwright is not None:
                await playwright.stop()

