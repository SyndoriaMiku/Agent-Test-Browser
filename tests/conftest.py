import pytest


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t
    def now(self) -> float:
        return self.t
    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()
