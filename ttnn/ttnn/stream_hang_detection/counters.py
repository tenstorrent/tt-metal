# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The counters that follow a chunk through each device of a pipeline stage, shared by the detector and the watcher.

A stage moves chunks through ttnn stream services. On every device, five counts in chunks follow a chunk:

    arrived     landed in an input: from the previous stage, or from the host
    consumed    the op that takes it in copied it out
    produced    the op that sends it handed it to an output's sender
    pushed      the sender handed it to fabric
    forwarded   the next stage took it

Each count is a 32-bit L1 word a service's kernels or workers bump, divided by how far the word moves per chunk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Word:
    """A 32-bit counter in a core's L1, and how far it moves per chunk."""

    x: int
    y: int
    addr: int  # address of the counter in L1
    unit: int  # how far the word moves per chunk
    raw: int = 0  # raw counter value
    wraps: int = 0  # number of times the counter has wrapped around

    @property
    def chunks(self) -> int:
        return (self.raw + (self.wraps << 32)) // self.unit


@dataclass
class InputCounters:
    """A stream into the device: H2D from the host, D2D from the previous stage."""

    name: str
    arrived: Word
    consumed: Word


@dataclass
class OutputCounters:
    """A stream out of the device. On a stage without outputs, its acks: all three are one counter."""

    name: str
    produced: Word
    pushed: Word
    forwarded: Word


@dataclass
class DeviceCounters:
    uid: int  # device id
    r: int  # row
    c: int  # column
    inputs: list[InputCounters]  # in the order the run loop receives them
    outputs: list[OutputCounters]  # in the order it sends them
    acks: Word | None  # D2H acks

    def words(self) -> list[Word]:
        """Every word, in the same order on every device of the stage."""
        words = [w for i in self.inputs for w in (i.arrived, i.consumed)]
        words += [w for o in self.outputs for w in (o.produced, o.pushed, o.forwarded)]
        return words if self.acks is None else words + [self.acks]

    def borrow_wraps(self, dev00: DeviceCounters) -> None:
        """Take device (0,0)'s wrap counts, one off where the two straddle a wrap.

        For a device read only at verdict time: a stage's devices exchange data throughout every chunk, so none
        gets more than a chunk or so away from device (0,0).
        """
        for w, w00 in zip(self.words(), dev00.words()):
            w.wraps = w00.wraps + round((w00.raw - w.raw) / (1 << 32))

    def counts(self) -> str:
        """This device's counts, in chunks."""
        parts = [f"{i.name} in arrived={i.arrived.chunks} consumed={i.consumed.chunks}" for i in self.inputs]
        parts += [
            f"{o.name} out produced={o.produced.chunks} pushed={o.pushed.chunks} forwarded={o.forwarded.chunks}"
            for o in self.outputs
        ]
        if self.acks is not None:
            parts.append(f"acks={self.acks.chunks}")
        return f"  device ({self.r},{self.c}): " + ", ".join(parts)
