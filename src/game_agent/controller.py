"""Browser controller stub: the async boundary that drives the game page.

Concrete input/output arrives in Milestone 1. The controller owns a held
input ledger: the keys/buttons currently held down on the page.
"""

from game_agent.observation import Observation
from game_agent.schema import Action


class BrowserController:
    """Async facade over the game page (Milestone 1).

    Tracks held keys/buttons internally: every hold leaves an entry in a
    private ledger so ``release_all`` can always unwind them.
    """

    async def open(self) -> None:
        """Launch the browser, load the game page, and get it playable.

        Raises:
            NotImplementedError: Until Milestone 1.
        """
        raise NotImplementedError("Milestone 1: BrowserController.open")

    async def screenshot(self) -> Observation:
        """Capture the viewport as an :class:`Observation`.

        Raises:
            NotImplementedError: Until Milestone 1.
        """
        raise NotImplementedError("Milestone 1: BrowserController.screenshot")

    async def execute(self, action: Action) -> None:
        """Dispatch a schema-valid :class:`Action` to the game page.

        Args:
            action: The validated action to perform.

        Raises:
            NotImplementedError: Until Milestone 1.
        """
        raise NotImplementedError("Milestone 1: BrowserController.execute")

    async def release_all(self) -> None:
        """Release every held key/button and empty the held-input ledger.

        Safe-recovery primitive the agent loop calls from ``RECOVERING``:
        a real implementation tracks what is held and releases each key or
        button so recovery never leaves inputs stuck down.

        Raises:
            NotImplementedError: Until Milestone 1.
        """
        raise NotImplementedError("Milestone 1: BrowserController.release_all")

    async def close(self) -> None:
        """Shut the browser down cleanly.

        Raises:
            NotImplementedError: Until Milestone 1.
        """
        raise NotImplementedError("Milestone 1: BrowserController.close")
