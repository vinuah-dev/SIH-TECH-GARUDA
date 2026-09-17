"""A tamper-evident chain over the event log.

An alert from a border post can end up as evidence. What matters then is not
only that the event was recorded, but that nobody edited or removed it
afterwards - and that this can be *shown*, not merely asserted by whoever
holds the database.

So every event is hashed together with the hash of the event before it. That
one link is what makes the log tamper-evident rather than tamper-proof:

* **Editing** any past event changes its hash, so the next event's recorded
  `prev_hash` no longer matches. The break points at the exact row.
* **Deleting** an event breaks the same link.
* **Appending** the attacker's own events is still possible - anyone with the
  database can extend the chain. What they cannot do is quietly change what is
  already in it.

That distinction is the honest one, and it is the difference between this and
a distributed ledger. A real chain of custody also needs the head hash
published somewhere the holder of the database does not control - read out at
shift handover, sent to a district server, or anchored on a public chain.
`head()` exists to make that a one-line integration; nothing here does it for
you, and the README says so.

The hash covers the event's stored JSON exactly as written, so verification
reads the same bytes an auditor would.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# The chain has to start somewhere. A fixed opening value means two posts
# built from the same events produce the same chain, which is what lets a
# district office compare them at all.
GENESIS = "0" * 64


def link(previous_hash: str, event_id: str, timestamp: str, payload: str) -> str:
    """The hash of one event, chained to the one before it.

    The fields are joined with a separator that cannot appear in a hex hash or
    an event id, so no two different events can be made to produce the same
    input string by moving a boundary - the classic way a naive concatenation
    is attacked.
    """
    material = "\x1f".join((previous_hash, event_id, timestamp, payload))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChainBreak:
    """Where the log stops agreeing with itself."""

    position: int
    event_id: str
    timestamp: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ChainReport:
    """The result of walking the whole chain."""

    events: int
    intact: bool
    head: str
    breaks: tuple[ChainBreak, ...] = ()

    def to_dict(self) -> dict:
        return {
            "events": self.events,
            "intact": self.intact,
            "head": self.head,
            "breaks": [b.to_dict() for b in self.breaks],
        }


def verify(rows) -> ChainReport:
    """Walk the log in write order and recompute every link.

    `rows` are mappings with event_id, timestamp, payload, prev_hash and hash,
    oldest first. Every break is reported rather than only the first: one
    edited row and one deleted row are different findings, and an auditor
    wants both.
    """
    previous = GENESIS
    breaks: list[ChainBreak] = []
    counted = 0

    for position, row in enumerate(rows):
        counted += 1
        event_id = row["event_id"]
        timestamp = row["timestamp"]
        stored_prev = row["prev_hash"]
        stored_hash = row["hash"]

        if stored_hash is None or stored_prev is None:
            breaks.append(ChainBreak(
                position, event_id, timestamp,
                "written before the chain existed, so it cannot be verified",
            ))
            # An unchained row cannot carry the chain forward; start again from
            # whatever it recorded rather than reporting every later row too.
            previous = stored_hash or previous
            continue

        if stored_prev != previous:
            breaks.append(ChainBreak(
                position, event_id, timestamp,
                "the previous event was changed or removed",
            ))

        expected = link(stored_prev, event_id, timestamp, row["payload"])
        if expected != stored_hash:
            breaks.append(ChainBreak(
                position, event_id, timestamp,
                "this event's own contents were changed after it was recorded",
            ))

        previous = stored_hash

    return ChainReport(
        events=counted,
        intact=not breaks,
        head=previous,
        breaks=tuple(breaks),
    )
