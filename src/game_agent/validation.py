"""Pure action validation: decide whether a decision may be executed.

``validate`` is a total function of ``(decision, snapshot, now, settings)``.
It only reads its arguments -- the snapshot, the settings, and the decision
are never mutated -- and it never reads the system clock: the caller passes
``now`` so validation stays deterministic and testable offline.

Checks run in a fixed order and the first rejection wins:
confidence, terminal-action bypass, coordinate bounds, allowlist, duration
guards, rate limit, action budget.
"""

from dataclasses import dataclass
from enum import StrEnum

from game_agent.config import Settings
from game_agent.schema import (
    Action,
    ClickAction,
    Decision,
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
from game_agent.session_state import StateSnapshot


class RejectedReason(StrEnum):
    """Why a decision was rejected."""

    OUT_OF_BOUNDS = "out_of_bounds"
    NOT_ALLOWED = "not_allowed"
    LIMIT_EXCEEDED = "limit_exceeded"
    RATE_LIMITED = "rate_limited"
    BUDGET_EXCEEDED = "budget_exceeded"
    LOW_CONFIDENCE = "low_confidence"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one decision."""

    accepted: bool
    reason: RejectedReason | None = None


def validate(
    decision: Decision,
    snapshot: StateSnapshot,
    now: float,
    settings: Settings,
) -> ValidationResult:
    """Validate one decision against the session snapshot and the settings.

    Returns an accepted result with no reason when the decision may be
    executed, otherwise a rejected result carrying the first rejection reason
    found.
    """
    action = decision.action

    if decision.confidence < settings.confidence_threshold:
        return _reject(RejectedReason.LOW_CONFIDENCE)

    # Terminal actions end or interrupt the session: they must still be
    # reportable, so they bypass the remaining checks.
    if isinstance(action, (DoneAction, UncertainAction)):
        return _accept()

    reason = _check_bounds(action, settings)
    if reason is None:
        reason = _check_allowlist(action, settings)
    if reason is None:
        reason = _check_durations(action, snapshot, settings)
    if reason is not None:
        return _reject(reason)

    if snapshot.last_action_at is not None and (
        now - snapshot.last_action_at < settings.min_action_interval_ms / 1000
    ):
        return _reject(RejectedReason.RATE_LIMITED)

    if snapshot.action_count >= settings.max_actions_per_session:
        return _reject(RejectedReason.BUDGET_EXCEEDED)

    return _accept()


def _accept() -> ValidationResult:
    """Build the accepted result."""
    return ValidationResult(accepted=True, reason=None)


def _reject(reason: RejectedReason) -> ValidationResult:
    """Build a rejected result carrying ``reason``."""
    return ValidationResult(accepted=False, reason=reason)


def _check_bounds(action: Action, settings: Settings) -> RejectedReason | None:
    """Check viewport bounds for coordinates and the wheel delta limit.

    Coordinates use a closed interval: ``x == viewport_width`` and
    ``y == viewport_height`` are inside the viewport.
    """
    if isinstance(action, WheelAction):
        if (
            abs(action.delta_x) > settings.max_wheel_delta
            or abs(action.delta_y) > settings.max_wheel_delta
        ):
            return RejectedReason.LIMIT_EXCEEDED
        return None

    for x, y in _coordinate_pairs(action):
        if not 0 <= x <= settings.viewport_width:
            return RejectedReason.OUT_OF_BOUNDS
        if not 0 <= y <= settings.viewport_height:
            return RejectedReason.OUT_OF_BOUNDS
    return None


def _coordinate_pairs(action: Action) -> tuple[tuple[int, int], ...]:
    """Return the viewport-relevant coordinate pairs of ``action``."""
    if isinstance(action, (ClickAction, MoveAction)):
        return ((action.x, action.y),)
    if isinstance(action, DragAction):
        return ((action.start_x, action.start_y), (action.end_x, action.end_y))
    return ()


def _check_allowlist(action: Action, settings: Settings) -> RejectedReason | None:
    """Check key/button targets against the configured allowlists.

    Every mouse button an action can target -- the button of a ``click`` or
    ``drag``, and the button of a ``hold``/``release`` -- must be listed in
    ``settings.allowed_buttons``. Buttons are compared by their string value
    (``schema.Button`` is a ``(str, Enum)`` mixin whose hash follows the enum
    member name, so the enum member itself would never match a set of
    strings).
    """
    if isinstance(action, PressAction):
        return None if action.key in settings.allowed_keys else RejectedReason.NOT_ALLOWED

    if isinstance(action, (HoldAction, ReleaseAction)):
        if action.key is not None:
            allowed = action.key in settings.allowed_keys
        elif action.button is not None:
            allowed = action.button.value in settings.allowed_buttons
        else:
            # Unreachable: the schema requires exactly one of key/button.
            allowed = False
        return None if allowed else RejectedReason.NOT_ALLOWED

    if isinstance(action, (ClickAction, DragAction)):
        allowed = action.button.value in settings.allowed_buttons
        return None if allowed else RejectedReason.NOT_ALLOWED

    return None


def _check_durations(
    action: Action,
    snapshot: StateSnapshot,
    settings: Settings,
) -> RejectedReason | None:
    """Check wait duration and accumulated hold time against their limits."""
    if isinstance(action, WaitAction):
        if action.duration_ms > settings.max_wait_ms:
            return RejectedReason.LIMIT_EXCEEDED
        return None

    if isinstance(action, HoldAction):
        if snapshot.accumulated_hold_ms + action.duration_ms > settings.max_hold_ms:
            return RejectedReason.LIMIT_EXCEEDED
        return None

    return None
