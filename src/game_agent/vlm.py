"""VLM configuration and client: observation plus goal to one decision.

The client returns structured decisions only; it never emits or runs
arbitrary code. Concrete model integration arrives in Milestone 2.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from dotenv import dotenv_values

from game_agent.observation import Observation
from game_agent.schema import Decision


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
