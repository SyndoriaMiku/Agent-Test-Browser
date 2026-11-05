from game_agent.config import Settings
from game_agent.schema import ClickAction, HoldAction
from game_agent.session_state import SessionState

def test_record_counts_and_budget(clock):
    s = SessionState(Settings(max_actions_per_session=2), clock=clock)
    assert s.remaining_action_budget() == 2
    s.record(ClickAction(type="click", x=1, y=1), clock.now())
    assert s.action_count == 1
    assert s.remaining_action_budget() == 1

def test_hold_accumulates(clock):
    s = SessionState(Settings(), clock=clock)
    s.record(HoldAction(type="hold", key="w", duration_ms=400), clock.now())
    s.record(HoldAction(type="hold", key="w", duration_ms=600), clock.now())
    assert s.accumulated_hold_ms == 1000

def test_failures_reset_on_success(clock):
    s = SessionState(Settings(), clock=clock)
    s.record_failure(); s.record_failure()
    assert s.consecutive_failures == 2
    s.record_success()
    assert s.consecutive_failures == 0

def test_snapshot_is_immutable_copy(clock):
    s = SessionState(Settings(), clock=clock)
    snap = s.snapshot()
    s.record(ClickAction(type="click", x=1, y=1), clock.now())
    assert snap.action_count == 0

def test_last_action_at_tracks_clock(clock):
    s = SessionState(Settings(), clock=clock)
    clock.advance(5.0)
    s.record(ClickAction(type="click", x=1, y=1), clock.now())
    assert s.last_action_at == 5.0

def test_default_clock_satisfies_clock_protocol():
    s = SessionState(Settings())
    assert isinstance(s._clock.now(), float)
