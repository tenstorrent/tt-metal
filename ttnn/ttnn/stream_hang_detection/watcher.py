# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Out-of-process watcher for one pipeline stage.

StreamHangDetector runs the file that defines the watcher's class, python <that file> <probe.json>, and the file
ends by calling that class's main(). This file does it for StreamHangWatcher itself; a pipeline's subclass overrides
what it needs and ends its own file the same way.

Poll the stage's counters on device (0,0); if none has moved for limit_s(), read every device once and judge(). On
a verdict it prints it and a stuck device's counts to stderr, runs the command for at most command_timeout_s, and
SIGKILLs the rank. With no command it exits instead and leaves the rank hung for inspection. By default a device of
a still stage waits on the first of:

    SENDER       an output's produced > pushed == forwarded: a ready chunk its sender never pushed
    OUTBOUND     an output's produced < the one before's produced == pushed: its send never ran
    COMPUTE      consumed > produced: took a chunk in and never produced it
    DOWNSTREAM   an output's produced > forwarded: the next stage has not taken our last chunk
    INBOUND      the next input's arrived > consumed: a chunk arrived and was never consumed
    UPSTREAM     anything else: idle, or the stage before us is stuck

A device has consumed a chunk once every input has, and produced it once the first output has. The next input is
the one the run loop waits on: the first with the lowest consumed. COMPUTE also covers the first output's outbound
op: both sit between consumed and produced.

The stage is the culprit if a device waits on SENDER, OUTBOUND or INBOUND, or on COMPUTE having produced fewer
chunks than every device consumed. Waiting on a neighbour is never the stage's fault as long as its run loop takes
a chunk in only after its previous push finished: then a stage held back by its downstream never has a chunk in hand
on every device.

These defaults assume a run loop that waits for its previous push, takes every input in, computes, then sends its
outputs one after another in list order. A pipeline whose run loop differs, or that moves chunks its counters do not
see, overrides state(), judge() or the counts they use.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto

from ttnn.stream_hang_detection.counters import DeviceCounters, InputCounters, OutputCounters, Word

MASK32 = 0xFFFFFFFF


@dataclass
class HangSettings:
    """When the watcher acts and what it runs then."""

    timeout_s: float  # how long device (0,0) may stand still
    first_chunk_timeout_s: float  # the same, until the first real chunk is out
    poll_s: float
    command: str  # run on a verdict, then the rank is SIGKILLed; empty leaves the rank hung
    command_timeout_s: float


class WaitingOn(Enum):
    """What a device of a still stage waits on; StreamHangWatcher.state takes the first that matches, in this order."""

    SENDER = auto()
    OUTBOUND = auto()
    COMPUTE = auto()
    DOWNSTREAM = auto()
    INBOUND = auto()
    UPSTREAM = auto()


# What the verdict says for each state that blames the stage, in the order it takes the blame.
BLAME = {
    WaitingOn.SENDER: "a sender never pushed a ready chunk",
    WaitingOn.OUTBOUND: "an outbound op never ran after the output before it went out",
    WaitingOn.COMPUTE: "never produced a chunk every device consumed",
    WaitingOn.INBOUND: "a chunk arrived and was never consumed",
}


@dataclass
class Judgement:
    """What the watcher makes of a stage's devices, read once while the stage stood still."""

    left_out: list[DeviceCounters]  # too far from device (0,0) to judge
    culprits: list[DeviceCounters]  # the devices the stage is to blame for, all in one state; empty when it is not
    blame: str = ""  # why, for the verdict line
    counts: str = ""  # a culprit's counts, printed under the verdict


@dataclass
class StreamHangWatcher:
    """One stage's watcher. A pipeline subclasses it and overrides what it needs; read() and run() stay as they are."""

    pid: int
    settings: HangSettings
    devices: list[DeviceCounters]  # devices[0] is device (0,0), the one polled

    def label(self) -> str:
        """What every line it prints starts with."""
        return f"STREAM HANG: pid {self.pid}"

    def limit_s(self, dev00: DeviceCounters) -> float:
        """How long device (0,0) may stand still before every device is read."""
        return self.settings.timeout_s if self.produced(dev00) >= 1 else self.settings.first_chunk_timeout_s

    def produced(self, device: DeviceCounters) -> int:
        return device.outputs[0].produced.chunks

    def consumed(self, device: DeviceCounters) -> int:
        return min(i.consumed.chunks for i in device.inputs)

    def state(self, device: DeviceCounters) -> WaitingOn:
        for o in device.outputs:
            if o.produced.chunks > o.pushed.chunks == o.forwarded.chunks:
                return WaitingOn.SENDER
        for before, o in zip(device.outputs, device.outputs[1:]):
            if o.produced.chunks < before.produced.chunks == before.pushed.chunks:
                return WaitingOn.OUTBOUND
        if self.consumed(device) > self.produced(device):
            return WaitingOn.COMPUTE
        # The next chunk is taken in only after the last push, so this waits on the next stage.
        if any(o.produced.chunks != o.forwarded.chunks for o in device.outputs):
            return WaitingOn.DOWNSTREAM
        nxt = min(device.inputs, key=lambda i: i.consumed.chunks)
        return WaitingOn.INBOUND if nxt.arrived.chunks > nxt.consumed.chunks else WaitingOn.UPSTREAM

    def counts(self, device: DeviceCounters) -> str:
        """The counts printed under a verdict."""
        return device.counts()

    def judge(self, devices: list[DeviceCounters]) -> Judgement:
        """Who is to blame, from one read of every device. devices[0] is device (0,0)."""
        dev00 = devices[0]
        # A word read from the wrong place must not convict: leave out devices more than a chunk off device (0,0).
        off = [
            d
            for d in devices
            if abs(self.consumed(d) - self.consumed(dev00)) > 1 or abs(self.produced(d) - self.produced(dev00)) > 1
        ]
        judged = [d for d in devices if d not in off]
        took = min(self.consumed(d) for d in judged)
        for on in BLAME:
            # Compute is judged across devices: a device that never got its share keeps the minimum down.
            hit = [
                d for d in judged if self.state(d) is on and (on is not WaitingOn.COMPUTE or self.produced(d) < took)
            ]
            if hit:
                return Judgement(left_out=off, culprits=hit, blame=BLAME[on], counts=self.counts(hit[0]))
        return Judgement(left_out=off, culprits=[])

    def read(self, device: DeviceCounters) -> bool:
        """Read every word; True if any moved. The counters are uint32: a word below its last read has wrapped."""
        # Imported here: the rank imports this class too, and only the watcher process needs tt-exalens.
        from ttexalens import tt_exalens_lib as lib

        moved = False
        for w in device.words():
            raw = lib.read_word_from_device(f"{w.x},{w.y}", w.addr, device_id=device.uid) & MASK32
            w.wraps += raw < w.raw
            moved |= raw != w.raw
            w.raw = raw
        return moved

    def run(self) -> int:
        dev00 = self.devices[0]
        frozen_since: float | None = None
        left_out: list[str] = []
        while True:
            try:
                # kill 0 is just a liveness probe badly named
                os.kill(self.pid, 0)
            except ProcessLookupError:
                return 0

            try:
                moved = self.read(dev00)
            except Exception as e:
                print(
                    f"{self.label()}: reading device (0,0) failed, no longer watching: {e!r}",
                    file=sys.stderr,
                    flush=True,
                )
                return 1

            now = time.monotonic()
            if moved:
                frozen_since = None
            elif frozen_since is None:
                frozen_since = now
            elif now - frozen_since >= self.limit_s(dev00):
                frozen_since = now
                # The rest of the stage stopped with device (0,0), so one read of each device shows where it stands.
                for device in self.devices[1:]:
                    try:
                        self.read(device)
                    except Exception as e:
                        print(
                            f"{self.label()}: reading device ({device.r},{device.c}) failed, no longer watching: {e!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                        return 1
                    device.borrow_wraps(dev00)
                judgement = self.judge(self.devices)
                coords = [f"({d.r},{d.c})" for d in judgement.left_out]
                if coords != left_out:
                    print(f"{self.label()}: left out devices {' '.join(coords)}", file=sys.stderr)
                    left_out = coords
                if judgement.culprits:
                    judged = len(self.devices) - len(judgement.left_out)
                    where = (
                        "every device"
                        if len(judgement.culprits) == judged
                        else "devices " + " ".join(f"({d.r},{d.c})" for d in judgement.culprits)
                    )
                    print(
                        f"{self.label()} is the culprit -- {judgement.blame}, on {where}\n{judgement.counts}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if not self.settings.command:
                        return 0
                    # Its own session, so a timeout ends everything the shell started, not just the shell.
                    command = subprocess.Popen(self.settings.command, shell=True, start_new_session=True)
                    try:
                        command.wait(timeout=self.settings.command_timeout_s)
                    except subprocess.TimeoutExpired:
                        print(
                            f"{self.label()}: command still running after {self.settings.command_timeout_s:g} s, "
                            "killed it",
                            file=sys.stderr,
                            flush=True,
                        )
                        os.killpg(command.pid, signal.SIGKILL)
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(self.pid, signal.SIGKILL)
                    return 1

            time.sleep(self.settings.poll_s)

    @classmethod
    def main(cls, probe_path: str) -> int:
        """The watcher process: rebuild the watcher StreamHangDetector wrote, then watch."""
        with open(probe_path) as f:
            d = json.load(f)
        d["settings"] = HangSettings(**d["settings"])
        d["devices"] = [
            DeviceCounters(
                uid=dev["uid"],
                r=dev["r"],
                c=dev["c"],
                inputs=[InputCounters(i["name"], Word(**i["arrived"]), Word(**i["consumed"])) for i in dev["inputs"]],
                outputs=[
                    OutputCounters(o["name"], Word(**o["produced"]), Word(**o["pushed"]), Word(**o["forwarded"]))
                    for o in dev["outputs"]
                ],
                acks=None if dev["acks"] is None else Word(**dev["acks"]),
            )
            for dev in d["devices"]
        ]
        # The rest, pid and a subclass's own fields, are as asdict wrote them from this class.
        watcher = cls(**d)
        os.unlink(probe_path)
        return watcher.run()


if __name__ == "__main__":
    sys.exit(StreamHangWatcher.main(sys.argv[1]))
