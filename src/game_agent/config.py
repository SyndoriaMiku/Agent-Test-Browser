"""Configuration: frozen Settings defaults plus GAME_AGENT_* environment overrides."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

_ENV_PREFIX = "GAME_AGENT_"

_DEFAULT_ALLOWED_KEYS = frozenset(
    {"w", "a", "s", "d", "up", "down", "left", "right", "space", "enter", "escape", "shift", "tab"}
)
_DEFAULT_ALLOWED_BUTTONS = frozenset({"left", "right", "middle"})


@dataclass(frozen=True)
class Settings:
    """Immutable agent configuration; all coordinates are CSS pixels."""

    viewport_width: int = 1280
    viewport_height: int = 720
    device_scale_factor: float = 1.0
    max_actions_per_session: int = 200
    min_action_interval_ms: int = 200
    max_wait_ms: int = 5000
    max_hold_ms: int = 2000
    max_wheel_delta: int = 500
    confidence_threshold: float = 0.55
    allowed_keys: frozenset[str] = field(default_factory=lambda: _DEFAULT_ALLOWED_KEYS)
    allowed_buttons: frozenset[str] = field(default_factory=lambda: _DEFAULT_ALLOWED_BUTTONS)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from defaults with GAME_AGENT_<FIELD_UPPER> overrides.

    Reads ``env`` when given (``env={}`` means no overrides, never falls back to
    the process environment), otherwise reads ``os.environ``. Only int/float
    fields are overridable via environment; the frozenset allowlists use their
    dataclass defaults (per-item allowlist configuration is deferred to M1).
    """
    source: Mapping[str, str] = os.environ if env is None else env
    overrides: dict[str, int | float] = {}
    for f in fields(Settings):
        raw = source.get(f"{_ENV_PREFIX}{f.name.upper()}")
        if raw is None:
            continue
        # Coerce by the default's type: int fields have int defaults,
        # float fields have float defaults; frozensets are not overridable.
        if isinstance(f.default, int):
            overrides[f.name] = int(raw)
        elif isinstance(f.default, float):
            overrides[f.name] = float(raw)
    return Settings(**overrides)
