"""Session state: action budget, hold accumulation, and failure streaks.

Time comes from an injected ``Clock`` so tests can advance it; production
uses a ``time.monotonic``-backed clock. Counters are read-only from outside; ``snapshot()``
returns an immutable copy for safe sharing.
"""

import time
from dataclasses import dataclass
from typing import Protocol

from game_agent.config import Settings
from game_agent.schema import Action, HoldAction


class Clock(Protocol):
    """Time source with a single monotonic ``now()`` in seconds."""

    def now(self) -> float: ...


class MonotonicClock:
    """Default ``Clock`` backed by ``time.monotonic``."""

    def now(self) -> float:
        return time.monotonic()


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable copy of the session counters at a point in time."""

    action_count: int
    accumulated_hold_ms: int
    consecutive_failures: int
    last_action_at: float | None


class SessionState:
    """Mutable per-session counters guarded by the configured budget."""

    def __init__(self, settings: Settings, clock: Clock | None = None) -> None:
        self._settings = settings
        self._clock: Clock = MonotonicClock() if clock is None else clock
        self._action_count = 0
        self._accumulated_hold_ms = 0
        self._consecutive_failures = 0
        self._last_action_at: float | None = None

    @property
    def action_count(self) -> int:
        return self._action_count

    @property
    def accumulated_hold_ms(self) -> int:
        return self._accumulated_hold_ms

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def last_action_at(self) -> float | None:
        return self._last_action_at

    def snapshot(self) -> StateSnapshot:
        """Return an immutable copy of the current counters."""
        return StateSnapshot(
            action_count=self._action_count,
            accumulated_hold_ms=self._accumulated_hold_ms,
            consecutive_failures=self._consecutive_failures,
            last_action_at=self._last_action_at,
        )

    def record(self, action: Action, now: float) -> None:
        """Record an executed action and its timestamp.

        Hold actions add ``duration_ms`` to ``accumulated_hold_ms``.
        """
        self._action_count += 1
        if isinstance(action, HoldAction):
            self._accumulated_hold_ms += action.duration_ms
        self._last_action_at = now

    def record_failure(self) -> None:
        """Extend the consecutive-failure streak by one."""
        self._consecutive_failures += 1

    def record_success(self) -> None:
        """Reset the consecutive-failure streak."""
        self._consecutive_failures = 0

    def remaining_action_budget(self) -> int:
        """Actions left before ``max_actions_per_session`` is exhausted."""
        return self._settings.max_actions_per_session - self._action_count
