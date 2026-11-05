import pytest
from game_agent.state import InvalidTransition, State, StateMachine

def test_happy_path():
    m = StateMachine()
    assert m.state == State.IDLE
    for s in [State.OBSERVING, State.DECIDING, State.VALIDATING, State.ACTING,
              State.VERIFYING, State.OBSERVING]:
        m.transition(s)
    assert m.state == State.OBSERVING

def test_invalid_transition_raises():
    m = StateMachine()
    with pytest.raises(InvalidTransition):
        m.transition(State.ACTING)

def test_done_is_reachable_from_validating():
    m = StateMachine()
    m.transition(State.OBSERVING); m.transition(State.DECIDING); m.transition(State.VALIDATING)
    m.transition(State.DONE)
    assert m.state == State.DONE

def test_needs_user_reachable_from_verifying_and_resumable():
    m = StateMachine()
    for s in [State.OBSERVING, State.DECIDING, State.VALIDATING, State.ACTING, State.VERIFYING]:
        m.transition(s)
    m.transition(State.NEEDS_USER)
    assert m.state == State.NEEDS_USER
    m.transition(State.OBSERVING)  # human resumed
    assert m.state == State.OBSERVING

def test_recovering_and_needs_user_have_exits():
    m = StateMachine()
    for s in [State.OBSERVING, State.DECIDING, State.VALIDATING, State.ACTING, State.RECOVERING]:
        m.transition(s)
    assert m.can(State.OBSERVING)
    assert m.can(State.NEEDS_USER)
    assert m.can(State.STOPPED)

def test_recovering_cannot_go_to_verifying():
    m = StateMachine()
    for s in [State.OBSERVING, State.DECIDING, State.VALIDATING, State.ACTING, State.RECOVERING]:
        m.transition(s)
    assert not m.can(State.VERIFYING)
    with pytest.raises(InvalidTransition):
        m.transition(State.VERIFYING)

def test_idle_can_reach_observing_and_stopped():
    m = StateMachine()
    assert m.can(State.OBSERVING)
    assert m.can(State.STOPPED)

def test_stop_is_terminal():
    m = StateMachine()
    m.stop()
    assert m.state == State.STOPPED
    with pytest.raises(InvalidTransition):
        m.transition(State.OBSERVING)
