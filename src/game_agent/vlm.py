"""VLM client stub: turns one observation plus goal into a single decision.

Concrete model integration arrives in Milestone 2. The client returns
structured decisions only; it never emits or runs arbitrary code.
"""

from game_agent.observation import Observation
from game_agent.schema import Decision


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
