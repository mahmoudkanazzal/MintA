"""Lifecycle state machines for experiment units and pipeline stages.

One transition table per lifecycle. Every status written to units.csv, results.csv,
runs/<run>/stages/*.json and the progress line comes from these machines; an invalid
transition raises.

Unit (one experiment unit, i.e. one row group of results.csv):
    pending -> running -> ok | failed;  failed -> running only while retries < MAX_RETRIES
Stage (data, experiments, aggregate):
    pending -> running -> outputs_written -> verified -> complete
"""

from __future__ import annotations

MAX_RETRIES = 2


class InvalidTransition(RuntimeError):
    """Raised when a lifecycle is asked to move along an edge its table does not allow."""


UNIT_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"running"}),
    "running": frozenset({"ok", "failed"}),
    "failed": frozenset({"running"}),
    "ok": frozenset(),
}
UNIT_TERMINAL = frozenset({"ok", "failed"})

STAGE_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"running"}),
    "running": frozenset({"outputs_written"}),
    "outputs_written": frozenset({"verified"}),
    "verified": frozenset({"complete"}),
    "complete": frozenset(),
}


class _Machine:
    table: dict[str, frozenset[str]]

    def __init__(self, name: str, state: str = "pending"):
        if state not in self.table:
            raise InvalidTransition(f"{name}: unknown state {state!r}")
        self.name = name
        self.state = state

    def to(self, new: str) -> str:
        if new not in self.table.get(self.state, frozenset()):
            raise InvalidTransition(f"{self.name}: {self.state} -> {new} is not allowed")
        self._check(new)
        self.state = new
        return new

    def _check(self, new: str) -> None:
        """Extra guard on an allowed edge (subclasses)."""


class UnitLifecycle(_Machine):
    table = UNIT_TRANSITIONS

    def __init__(self, name: str, max_retries: int = MAX_RETRIES):
        super().__init__(name)
        if not 0 <= max_retries <= MAX_RETRIES:
            raise ValueError(f"max_retries must be in [0, {MAX_RETRIES}], got {max_retries}")
        self.max_retries = max_retries
        self.retries = 0

    def _check(self, new: str) -> None:
        if self.state == "failed" and new == "running":
            if self.retries >= self.max_retries:
                raise InvalidTransition(f"{self.name}: failed -> running refused after {self.retries} retries")
            self.retries += 1

    @property
    def terminal(self) -> bool:
        return self.state in UNIT_TERMINAL


class StageLifecycle(_Machine):
    table = STAGE_TRANSITIONS


def recorded_unit_status(status: str) -> str:
    """Validate a status read back from units.csv: only terminal unit states are recorded."""
    if status not in UNIT_TERMINAL:
        raise InvalidTransition(f"units.csv holds non-terminal status {status!r}")
    return status
