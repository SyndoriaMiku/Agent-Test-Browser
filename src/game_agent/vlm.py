"""VLM configuration and client: observation plus goal to one decision.

The client returns structured decisions only; it never emits or runs
arbitrary code. Concrete model integration arrives in Milestone 2.
"""

import base64
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from game_agent.observation import Observation
from game_agent.schema import Decision, UncertainAction


class VLMError(Exception):
    """Raised when a VLM call fails or its output cannot be used."""


@dataclass(frozen=True)
class VLMConfig:
    """Settings for the OpenAI-compatible vision-language model endpoint."""

    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str | None = None
    api_key: str | None = None
    timeout_s: float = 60.0
    max_tokens: int = 512
    temperature: float = 0.0


def _number(
    env: Mapping[str, str],
    key: str,
    parse: Callable[[str], float],
    default: float,
) -> float:
    """Read numeric option *key* from *env*, falling back to *default*."""
    raw = env.get(key)
    return default if raw is None else parse(raw)


def load_vlm_config(env: Mapping[str, str] | None = None) -> VLMConfig:
    """Build :class:`VLMConfig` from ``NVIDIA_*`` variables.

    Args:
        env: Explicit mapping to read. When given (even empty) it is the
            only source: neither ``.env`` nor the process environment is
            consulted. When ``None``, ``.env`` values are merged beneath
            the process environment (process wins); ``os.environ`` is
            never mutated.

    Returns:
        The resolved configuration. ``api_key`` and ``model`` have no
        defaults and stay ``None`` unless ``NVIDIA_API_KEY`` /
        ``NVIDIA_MODEL`` provide them.
    """
    if env is None:
        env = {**dotenv_values(), **os.environ}
    return VLMConfig(
        base_url=env.get("NVIDIA_BASE_URL", VLMConfig.base_url),
        model=env.get("NVIDIA_MODEL"),
        api_key=env.get("NVIDIA_API_KEY"),
        timeout_s=_number(env, "NVIDIA_TIMEOUT_S", float, VLMConfig.timeout_s),
        max_tokens=_number(env, "NVIDIA_MAX_TOKENS", int, VLMConfig.max_tokens),
        temperature=_number(env, "NVIDIA_TEMPERATURE", float, VLMConfig.temperature),
    )


_DECISION_SCHEMA_JSON: str = json.dumps(Decision.model_json_schema(), indent=2)
"""Exact JSON schema of :class:`Decision`, rendered once for the prompt."""

_SYSTEM_PROMPT = f"""\
You are the decision-making brain of a browser game agent. You see one
screenshot of the current viewport and must choose the next action.

Reply with ONLY one JSON object matching this exact JSON schema for a
Decision (no prose, no Markdown fence, no extra keys):

{_DECISION_SCHEMA_JSON}

Available action types (the discriminating "type" field):
- click: press a mouse button at (x, y); optional "button"
  ("left"|"right"|"middle") and "click_count" (1-3).
- move: move the pointer to (x, y).
- press: press and release one keyboard key.
- hold: hold a key or mouse button down; requires "duration_ms" > 0 and
  exactly one of "key" or "button".
- release: release a held key or button; exactly one of "key" or "button".
- drag: drag from (start_x, start_y) to (end_x, end_y) over "duration_ms"
  > 0 with a held mouse button.
- wheel: scroll the mouse wheel by "delta_x"/"delta_y" pixels.
- wait: do nothing for "duration_ms" > 0.
- done: the goal is met; include a short non-empty "summary".
- uncertain: the screenshot cannot be interpreted confidently; include a
  short non-empty "reason".

Rules:
- All coordinates are CSS pixels relative to the viewport size given in
  the user message; the top-left corner is (0, 0).
- "hold" always needs "duration_ms".
- "state" is a short label for the current screen (for example "menu",
  "game", "game_over").
- Use "uncertain" when you are not sure what the screen shows.
- Use "done" only when the goal is fully met.
- Output ONLY one JSON object: no prose, no explanation, no code fence.
"""


def encode_image_data_url(path: str | Path) -> str:
    """Read the PNG at *path* and return it as a base64 data URL.

    Args:
        path: Path to the screenshot file.

    Returns:
        A ``data:image/png;base64,<...>`` URL.

    Raises:
        OSError: If the file cannot be read.
    """
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_messages(
    image_data_url: str,
    goal: str,
    remaining_actions: int,
    viewport_width: int,
    viewport_height: int,
) -> list[dict]:
    """Build the chat messages sent to the model.

    Args:
        image_data_url: The screenshot as a data URL (see
            :func:`encode_image_data_url`).
        goal: The player-facing goal text.
        remaining_actions: Actions left in the session budget.
        viewport_width: Viewport width in CSS pixels.
        viewport_height: Viewport height in CSS pixels.

    Returns:
        A system message (schema, action list, rules) and a user message
        whose ``content`` is a text part followed by the image part.
    """
    text = (
        f"Goal: {goal}\n"
        f"Remaining actions: {remaining_actions}\n"
        f"Viewport: {viewport_width}x{viewport_height} CSS pixels.\n"
        "The screenshot follows."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def _strip_code_fence(text: str) -> str:
    """Strip whitespace and one optional Markdown code fence.

    The opening fence may carry a language tag such as ``json``.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_decision(content: str, observation_id: str) -> Decision:
    """Parse model output into a :class:`Decision`.

    Minimal normalization only: whitespace and one optional code fence
    are stripped; types are never coerced and unknown keys are never
    dropped (the schema forbids them).

    Args:
        content: Raw model output.
        observation_id: Observation the decision is based on; overwrites
            whatever ``observation_id`` the model returned.

    Returns:
        The validated decision with *observation_id* set.

    Raises:
        pydantic.ValidationError: If the output is not a schema-valid
            ``Decision``.
        ValueError: If the output is not valid JSON at all.
    """
    payload = _strip_code_fence(content)
    decision = Decision.model_validate_json(payload)
    return decision.model_copy(update={"observation_id": observation_id})


def fallback_decision(observation_id: str, error: str) -> Decision:
    """Build the terminal ``uncertain`` decision for unparseable output.

    Used after the one repair attempt also failed. ``confidence=1.0`` is
    intentional: the validator checks confidence before bypassing
    terminal actions, so a low value would never surface as
    "screen could not be read".

    Args:
        observation_id: Observation the failed decision was based on.
        error: Description of the parse failure (kept out of the
            decision itself).

    Returns:
        The exact fallback decision.
    """
    return Decision(
        state="uncertain",
        reason="model output was not valid JSON after one repair",
        action=UncertainAction(type="uncertain", reason="unparseable model output"),
        expected_result=None,
        confidence=1.0,
        observation_id=observation_id,
    )


class VLMClient:
    """Async facade over the vision-language model (Milestone 2)."""

    async def decide(
        self,
        observation: Observation,
        goal: str,
        remaining_actions: int,
    ) -> Decision:
        """Ask the model for the next :class:`Decision`.

        Must return a schema-valid :class:`Decision` (validated against the
        action schema) and never arbitrary code — unparseable model output
        is an error, not something to execute.

        Args:
            observation: The observation the decision is based on.
            goal: The player-facing goal text.
            remaining_actions: Actions left in the session budget.

        Returns:
            The schema-valid decision for this observation.

        Raises:
            NotImplementedError: Until Milestone 2.
        """
        raise NotImplementedError("Milestone 2: VLMClient.decide")
