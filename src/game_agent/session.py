"""Closed-loop session: observe -> decide -> validate -> act -> verify.

``Session`` drives one agent session over an already-open controller and a
vision-language model.  Every cycle walks exactly the lifecycle the
``StateMachine`` table allows -- ``IDLE -> OBSERVING -> DECIDING ->
VALIDATING -> ACTING -> VERIFYING -> OBSERVING`` -- and every exit path
runs ``controller.release_all()``, so no key or button is left held down.

Time and sleeping are injected (``clock``/``sleep``) so the loop is
deterministic in tests.  The minimum interval between two actions is waited
out *before* the decision is validated: the validator would otherwise reject
the action as ``RATE_LIMITED`` because the last action was too recent.
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from game_agent.config import Settings
from game_agent.logging import SessionLogger
from game_agent.observation import Observation
from game_agent.overlay import add_coordinate_grid
from game_agent.schema import Action, Decision, DoneAction, UncertainAction
from game_agent.session_state import Clock, MonotonicClock, SessionState
from game_agent.state import State, StateMachine
from game_agent.validation import ValidationResult, validate

_PACE_MARGIN_S = 1e-6
"""Seconds slept on top of the remaining interval.

The validator rejects an action when less than ``min_action_interval_ms``
passed since the previous one.  Floating-point rounding can leave the
interval a fraction of a microsecond short, so the pace sleeps a hair
longer and the wait is never borderline.
"""


class ControllerProtocol(Protocol):
    """The part of the browser controller a session drives."""

    async def screenshot(self) -> Observation: ...

    async def execute(self, action: Action) -> None: ...

    async def release_all(self) -> None: ...


class VLMProtocol(Protocol):
    """The part of the vision model a session asks."""

    async def decide(
        self, observation: Observation, goal: str, remaining_actions: int
    ) -> Decision: ...


@dataclass(frozen=True)
class SessionResult:
    """How a session ended.

    ``state`` is the lifecycle state the session finished in, ``cycles`` the
    number of cycles that ran, ``reason`` a short explanation of the end,
    and ``last_decision`` the last decision the model made (``None`` when
    the model never got to decide).
    """

    state: State
    cycles: int
    reason: str
    last_decision: Decision | None


def _screenshot_path(item: Observation | str | Path | None) -> str | None:
    """Return the screenshot path of *item* (an observation or a path)."""
    if isinstance(item, Observation):
        return item.screenshot_path
    return None if item is None else str(item)


def _image_changed(
    before: Observation | str | Path | None,
    after: Observation | str | Path | None,
) -> bool:
    """Whether the screen changed between two screenshots.

    The two PNG files are compared byte-for-byte: Playwright renders an
    unchanged screen to the same bytes, so any difference means the screen
    changed.  A missing screenshot counts as changed -- the session cannot
    prove otherwise, and calling it unchanged would hide a broken capture.
    """
    before_path = _screenshot_path(before)
    after_path = _screenshot_path(after)
    if before_path is None or after_path is None:
        return True
    try:
        return Path(before_path).read_bytes() != Path(after_path).read_bytes()
    except OSError:
        return True


class Session:
    """One closed-loop agent session.

    The controller must already be open; the session never opens or closes
    the browser.  It owns the lifecycle (``StateMachine``), the counters
    (``SessionState``), and the optional JSONL log.
    """

    def __init__(
        self,
        controller: ControllerProtocol,
        vlm: VLMProtocol,
        settings: Settings,
        *,
        logger: SessionLogger | None = None,
        max_consecutive_failures: int = 3,
        repeat_limit: int = 3,
        grid_step: int | None = 100,
        clock: Clock | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """Store the collaborators and the session limits.

        Args:
            controller: The browser controller to drive.
            vlm: The vision model that turns an observation into a decision.
            settings: Viewport geometry, policy limits and budgets.
            logger: Optional JSONL session log; without it nothing is logged.
            max_consecutive_failures: Failures tolerated before the session
                asks the user for help.
            repeat_limit: How often the same action may repeat in a row
                before the session asks the user for help.
            grid_step: Spacing (CSS pixels) of the coordinate grid drawn onto
                the screenshot the model sees; ``None`` disables the grid.
            clock: Time source; defaults to a monotonic clock.
            sleep: Awaitable sleep; defaults to :func:`asyncio.sleep`.
        """
        self._controller = controller
        self._vlm = vlm
        self._settings = settings
        self._logger = logger
        self._max_consecutive_failures = max_consecutive_failures
        self._repeat_limit = repeat_limit
        self._grid_step = grid_step
        self._clock: Clock = MonotonicClock() if clock is None else clock
        self._sleep: Callable[[float], Awaitable[None]] = (
            asyncio.sleep if sleep is None else sleep
        )
        self._state = SessionState(settings, clock=self._clock)
        self._machine = StateMachine()
        self._stopped = False
        self._last_decision: Decision | None = None

    async def run(self, goal: str, max_cycles: int = 50) -> SessionResult:
        """Run the loop until the session ends.

        The loop ends when the model reports the goal done, when it cannot
        act safely, when the cycle limit is reached, or when :meth:`stop` is
        called.  Whatever ends it, ``controller.release_all()`` runs before
        ``run`` returns.

        Args:
            goal: The player-facing goal the model is asked to achieve.
            max_cycles: Upper bound on the number of cycles to run.

        Returns:
            The :class:`SessionResult` describing how the session ended.
        """
        cycles = 0
        reason = ""
        try:
            while cycles < max_cycles:
                if self._stopped:
                    self._machine.transition(State.STOPPED)
                    reason = "stop requested"
                    break

                cycles += 1
                self._machine.transition(State.OBSERVING)
                await self._pace()
                obs = await self._controller.screenshot()
                vlm_obs = self._with_grid(obs)
                self._log_observation(obs)

                self._machine.transition(State.DECIDING)
                decision = await self._vlm.decide(
                    vlm_obs, goal, self._state.remaining_action_budget()
                )
                self._last_decision = decision
                self._log_decision(obs, decision)

                self._machine.transition(State.VALIDATING)
                validation = validate(
                    decision,
                    self._state.snapshot(),
                    self._clock.now(),
                    self._settings,
                )
                if not validation.accepted:
                    self._machine.transition(State.NEEDS_USER)
                    reason = f"decision rejected ({validation.reason})"
                    break

                action = decision.action
                if isinstance(action, DoneAction):
                    self._machine.transition(State.DONE)
                    reason = "goal reported complete"
                    break
                if isinstance(action, UncertainAction):
                    self._machine.transition(State.NEEDS_USER)
                    reason = f"model is uncertain ({action.reason})"
                    break

                self._machine.transition(State.ACTING)
                await self._controller.execute(action)
                self._state.record(action, self._clock.now())
                self._log_action(obs, action, validation)

                self._machine.transition(State.VERIFYING)
                after = await self._controller.screenshot()
                changed = _image_changed(obs.screenshot_path, after.screenshot_path)
                self._log_verify(obs, changed)
                if changed:
                    self._state.record_success()
            else:
                # No branch broke out of the loop: the cycle limit ended it.
                self._machine.transition(State.STOPPED)
                reason = f"cycle limit reached ({max_cycles} cycles)"
        finally:
            await self._release_all()

        return SessionResult(
            state=self._machine.state,
            cycles=cycles,
            reason=reason,
            last_decision=self._last_decision,
        )

    def stop(self) -> None:
        """Ask the session to stop at the next cycle boundary."""
        self._stopped = True

    async def _pace(self) -> None:
        """Wait out the minimum interval since the last executed action."""
        last_action_at = self._state.last_action_at
        if last_action_at is None:
            return
        interval = self._settings.min_action_interval_ms / 1000
        elapsed = self._clock.now() - last_action_at
        if elapsed < interval:
            await self._sleep(interval - elapsed + _PACE_MARGIN_S)

    def _with_grid(self, obs: Observation) -> Observation:
        """Return the observation the model should see.

        The coordinate grid is drawn onto a copy of the screenshot, so the
        original capture stays untouched and the observation keeps its id.
        """
        if self._grid_step is None or obs.screenshot_path is None:
            return obs
        grid_path = add_coordinate_grid(
            obs.screenshot_path,
            step=self._grid_step,
            scale=obs.device_scale_factor,
        )
        return replace(obs, screenshot_path=grid_path)

    async def _release_all(self) -> None:
        """Release every held input; a failure here must not mask the result."""
        with contextlib.suppress(Exception):
            await self._controller.release_all()

    def _log_observation(self, obs: Observation) -> None:
        """Record the captured observation."""
        if self._logger is not None:
            self._logger.log_observation(obs)

    def _log_decision(self, obs: Observation, decision: Decision) -> None:
        """Record the decision made for *obs*."""
        if self._logger is not None:
            self._logger.log_decision(obs, decision)

    def _log_action(
        self, obs: Observation, action: Action, result: ValidationResult
    ) -> None:
        """Record the executed action and its validation outcome."""
        if self._logger is not None:
            self._logger.log_action(obs, action, result)

    def _log_verify(self, obs: Observation, changed: bool) -> None:
        """Record whether the screen changed after the action."""
        if self._logger is not None:
            self._logger.log_verify(
                obs, changed, "screen changed" if changed else "screen unchanged"
            )
