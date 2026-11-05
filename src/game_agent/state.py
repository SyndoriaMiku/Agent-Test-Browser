"""State machine: guarded transitions over the agent lifecycle.

States form a table-driven lifecycle starting at ``IDLE``; every move is
validated against the transition table and illegal moves raise
``InvalidTransition``. ``STOPPED`` is terminal. ``note_failure()`` /
``note_success()`` track a consecutive-failure streak so the caller can route
to ``NEEDS_USER`` once ``max_consecutive_failures`` is reached.
"""

from enum import Enum, auto


class State(Enum):
    """Lifecycle states of the agent."""

    IDLE = auto()
    OBSERVING = auto()
    DECIDING = auto()
    VALIDATING = auto()
    ACTING = auto()
    VERIFYING = auto()
    RECOVERING = auto()
    DONE = auto()
    NEEDS_USER = auto()
    STOPPED = auto()


class InvalidTransition(Exception):
    """Raised when a transition is not allowed from the current state."""

    def __init__(self, target: State) -> None:
        self.target = target
        super().__init__(f"transition to {target.name} is not allowed")


_TRANSITIONS: dict[State, frozenset[State]] = {
    State.IDLE: frozenset({State.OBSERVING, State.STOPPED}),
    State.OBSERVING: frozenset({State.DECIDING, State.NEEDS_USER, State.STOPPED}),
    State.DECIDING: frozenset({State.VALIDATING, State.NEEDS_USER, State.STOPPED}),
    State.VALIDATING: frozenset(
        {State.ACTING, State.OBSERVING, State.DONE, State.NEEDS_USER, State.STOPPED}
    ),
    State.ACTING: frozenset({State.VERIFYING, State.RECOVERING, State.STOPPED}),
    State.VERIFYING: frozenset(
        {State.OBSERVING, State.DONE, State.RECOVERING, State.NEEDS_USER, State.STOPPED}
    ),
    State.RECOVERING: frozenset(
        {State.OBSERVING, State.VERIFYING, State.NEEDS_USER, State.STOPPED}
    ),
    State.DONE: frozenset({State.STOPPED}),
    State.NEEDS_USER: frozenset({State.OBSERVING, State.STOPPED}),
    State.STOPPED: frozenset(),
}


class StateMachine:
    """Lifecycle state machine guarded by the transition table."""

    def __init__(self, max_consecutive_failures: int = 3) -> None:
        self._max_consecutive_failures = max_consecutive_failures
        self._state = State.IDLE
        self._consecutive_failures = 0

    @property
    def state(self) -> State:
        return self._state

    def can(self, target: State) -> bool:
        """Whether ``transition(target)`` would be allowed right now."""
        return target in _TRANSITIONS[self._state]

    def transition(self, target: State) -> None:
        """Move to ``target`` if the table allows it, else raise ``InvalidTransition``."""
        if not self.can(target):
            raise InvalidTransition(target)
        self._state = target

    def note_failure(self) -> None:
        """Extend the consecutive-failure streak by one."""
        self._consecutive_failures += 1

    def note_success(self) -> None:
        """Reset the consecutive-failure streak."""
        self._consecutive_failures = 0

    def stop(self) -> None:
        """Force the machine into the terminal ``STOPPED`` state."""
        self._state = State.STOPPED
