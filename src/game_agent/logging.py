"""JSONL session logger: one JSON object per line, LF-terminated.

Each record carries ``event``, ``timestamp``, and ``observation_id`` and
contains only data derived from the ``Observation``/``Decision``/``Action``/
``ValidationResult`` passed in -- never environment details, tokens, or auth
headers. Time comes from an injected ``Clock`` (default ``MonotonicClock``)
so logging stays deterministic in tests.
"""

import json
from pathlib import Path

from game_agent.observation import Observation
from game_agent.schema import Action, Decision
from game_agent.session_state import Clock, MonotonicClock
from game_agent.validation import ValidationResult


class SessionLogger:
    """Append-only JSONL writer for one agent session."""

    def __init__(self, path: str | Path, clock: Clock | None = None) -> None:
        self._path = Path(path)
        self._clock: Clock = MonotonicClock() if clock is None else clock
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Line buffering keeps each record on disk as it is written.
        self._file = self._path.open("a", encoding="utf-8", newline="\n", buffering=1)

    def log_observation(self, obs: Observation) -> None:
        """Record that an observation was captured."""
        self._write(
            {
                "event": "observation",
                "timestamp": self._clock.now(),
                "observation_id": obs.observation_id,
            }
        )

    def log_decision(self, obs: Observation, decision: Decision) -> None:
        """Record the decision made for an observation."""
        self._write(
            {
                "event": "decision",
                "timestamp": self._clock.now(),
                "observation_id": obs.observation_id,
                "decision": decision.model_dump(mode="json"),
            }
        )

    def log_action(
        self,
        obs: Observation,
        action: Action,
        result: ValidationResult | None = None,
    ) -> None:
        """Record the action executed for an observation, with its validation
        outcome when one is given."""
        record: dict[str, object] = {
            "event": "action",
            "timestamp": self._clock.now(),
            "observation_id": obs.observation_id,
            "action": action.model_dump(mode="json"),
        }
        if result is not None:
            record["accepted"] = result.accepted
            record["reason"] = result.reason
        self._write(record)

    def close(self) -> None:
        """Flush and close the log file."""
        if not self._file.closed:
            self._file.close()

    def _write(self, record: dict[str, object]) -> None:
        """Serialize one record as a single LF-terminated JSON line."""
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
