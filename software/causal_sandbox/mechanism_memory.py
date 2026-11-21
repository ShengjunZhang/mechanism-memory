"""Identity-indexed tuple replacement, conditional on supplied alignment."""
from __future__ import annotations

from typing import Hashable, NamedTuple


class MechanismTuple(NamedTuple):
    action: str
    context: Hashable
    target: str
    value: Hashable
    delay: int

    @property
    def lookup_key(self):
        return self.action, self.context, self.target, self.delay


class IdentityMemory:
    def __init__(self, entries: dict[Hashable, MechanismTuple] | None = None):
        self.entries = dict(entries or {})
        self.index: dict[tuple, set[Hashable]] = {}
        for identity, entry in self.entries.items():
            self.index.setdefault(entry.lookup_key, set()).add(identity)

    def replace(self, identity: Hashable, entry: MechanismTuple) -> None:
        previous = self.entries.get(identity)
        if previous is not None:
            owners = self.index[previous.lookup_key]
            owners.remove(identity)
            if not owners:
                del self.index[previous.lookup_key]
        self.entries[identity] = entry
        self.index.setdefault(entry.lookup_key, set()).add(identity)

    def local_update(self, replacements: dict[Hashable, MechanismTuple]) -> None:
        if not replacements.keys() <= self.entries.keys():
            raise KeyError("Local replacement requires aligned existing identities")
        for identity, entry in replacements.items():
            self.replace(identity, entry)

    def tuples(self) -> set[MechanismTuple]:
        return set(self.entries.values())
