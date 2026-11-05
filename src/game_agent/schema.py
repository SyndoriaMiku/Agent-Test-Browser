"""Action schema: discriminated action union and the frozen Decision model.

Structural bounds only (click_count range, positive durations, confidence in
[0.0, 1.0], non-empty strings); viewport and allowlist checks belong to the
validator, not the schema.
"""

from enum import Enum
from typing import Annotated, Literal, Self, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Button(str, Enum):
    """Mouse buttons an action can target."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class _Base(BaseModel):
    """Shared config: immutable models that reject unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class _KeyOrButton(_Base):
    """A target that is exactly one of a keyboard key or a mouse button."""

    key: str | None = Field(default=None, min_length=1)
    button: Button | None = None

    @model_validator(mode="after")
    def _validate_target(self) -> Self:
        if (self.key is None) == (self.button is None):
            raise ValueError("exactly one of 'key' or 'button' is required")
        return self


class ClickAction(_Base):
    """Click a mouse button at coordinates."""

    type: Literal["click"] = "click"
    x: int
    y: int
    button: Button = Button.LEFT
    click_count: int = Field(default=1, ge=1, le=3)


class MoveAction(_Base):
    """Move the pointer to coordinates."""

    type: Literal["move"] = "move"
    x: int
    y: int


class PressAction(_Base):
    """Press and release a keyboard key."""

    type: Literal["press"] = "press"
    key: str = Field(min_length=1)


class HoldAction(_KeyOrButton):
    """Hold a key or button down for a duration."""

    type: Literal["hold"] = "hold"
    duration_ms: int = Field(gt=0)


class ReleaseAction(_KeyOrButton):
    """Release a held key or button."""

    type: Literal["release"] = "release"


class DragAction(_Base):
    """Drag from one point to another with a mouse button held."""

    type: Literal["drag"] = "drag"
    from_x: int
    from_y: int
    to_x: int
    to_y: int
    button: Button = Button.LEFT


class WheelAction(_Base):
    """Scroll the mouse wheel by pixel deltas."""

    type: Literal["wheel"] = "wheel"
    dx: int = 0
    dy: int = 0


class WaitAction(_Base):
    """Wait for a duration."""

    type: Literal["wait"] = "wait"
    duration_ms: int = Field(gt=0)


class DoneAction(_Base):
    """Report the task finished."""

    type: Literal["done"] = "done"
    summary: str = Field(min_length=1)


class UncertainAction(_Base):
    """Report that the screen cannot be interpreted confidently."""

    type: Literal["uncertain"] = "uncertain"
    summary: str = Field(min_length=1)


Action = Annotated[
    Union[
        ClickAction,
        MoveAction,
        PressAction,
        HoldAction,
        ReleaseAction,
        DragAction,
        WheelAction,
        WaitAction,
        DoneAction,
        UncertainAction,
    ],
    Field(discriminator="type"),
]


class Decision(_Base):
    """One agent decision: what it observed, chose, and expects."""

    state: str
    reason: str = Field(min_length=1)
    action: Action
    expected_result: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    observation_id: str | None = None
