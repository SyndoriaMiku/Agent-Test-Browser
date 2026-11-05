"""Observation value type: one captured screen state with its metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """Immutable record of a single observation pass."""

    observation_id: str
    screenshot_path: str | None
    viewport_width: int
    viewport_height: int
    device_scale_factor: float
    timestamp: float
