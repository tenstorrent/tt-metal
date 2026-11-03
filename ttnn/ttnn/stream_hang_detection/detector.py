# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""In-process side of the stream hang watcher: describe a stage's counters, then hand the watcher to its own process.

describe_devices lists, for every device of the stage, the counters that follow a chunk through it, as each stream
service reports them: where each lives (core and L1 address) and how far it moves per chunk. A pipeline builds its
watcher with them; StreamHangDetector writes that watcher to a file as one line of JSON and runs the file that
defines its class, which polls the counters out of process. Nothing here touches them after that. Spread out, the
hand-off of PrefillStreamHangWatcher on a GLM-5.3 middle rank, with one device (addresses made up):

    {"pid": 4242,
     "settings": {"timeout_s": 10.0, "first_chunk_timeout_s": 100.0, "poll_s": 2.0, "command": "",
                  "command_timeout_s": 300.0},
     "devices": [
       {"uid": 81985529, "r": 0, "c": 0,
        "inputs": [{"name": "activation",
                    "arrived": {"x": 1, "y": 9, "addr": 1024, "unit": 2, "raw": 0, "wraps": 0},
                    "consumed": {"x": 1, "y": 9, "addr": 1088, "unit": 64, "raw": 0, "wraps": 0}},
                   {"name": "indices", "arrived": {...}, "consumed": {...}}],
        "outputs": [{"name": "activation", "produced": {...}, "pushed": {...}, "forwarded": {...}},
                    {"name": "indices", "produced": {...}, "pushed": {...}, "forwarded": {...}}],
        "acks": {"x": 3, "y": 9, "addr": 2048, "unit": 1, "raw": 0, "wraps": 0}},
       ...],
     "stage_idx": 1, "layers": "20-39", "warmup_out": 1, "warmup_in": 1}
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import ttnn
from ttnn.stream_hang_detection.counters import DeviceCounters, InputCounters, OutputCounters, Word
from ttnn.stream_hang_detection.watcher import StreamHangWatcher


@dataclass
class Stream:
    """One stream service of a stage, under the name the watcher prints its counts with."""

    name: str
    service: ttnn.H2DStreamService | ttnn.D2DStreamServiceReceiver | ttnn.D2DStreamServiceSender


@dataclass
class AckStream:
    """A stage's D2H stream, whose write acks the watcher counts."""

    name: str
    service: ttnn.D2HStreamService
    acks_per_chunk: int  # how many acks one chunk sends


def _word(counter: ttnn.StreamCounter) -> Word:
    return Word(x=counter.core.x, y=counter.core.y, addr=counter.addr, unit=counter.step)


def _input(name: str, counters: ttnn.StreamInputCounters) -> InputCounters:
    return InputCounters(name, _word(counters.arrived), _word(counters.consumed))


def _output(name: str, counters: ttnn.StreamOutputCounters) -> OutputCounters:
    return OutputCounters(name, _word(counters.produced), _word(counters.pushed), _word(counters.forwarded))


def describe_devices(
    mesh_device: ttnn.MeshDevice,
    *,
    inputs: list[Stream],
    outputs: list[Stream],
    acks: AckStream | None = None,
) -> list[DeviceCounters] | None:
    """Every device of the stage, with its streams' counters. None without outputs or acks: nothing would say a chunk
    finished, so the stage can't be watched.

    inputs and outputs: in the order the run loop receives and sends them, the activation first.
    acks: the finished-chunk word of a stage without outputs and a heartbeat on the others.
    """
    if not outputs and acks is None:
        return None
    devices = []
    for r in range(mesh_device.shape[0]):
        for c in range(mesh_device.shape[1]):
            coord = ttnn.MeshCoordinate(r, c)
            node = mesh_device.get_fabric_node_id(coord)
            acks_word = None if acks is None else _word(acks.service.get_write_ack_counter(coord))
            device_outputs = [_output(s.name, s.service.get_output_counters(coord)) for s in outputs]
            if not device_outputs:
                # A chunk is out once all its acks are in, with no sender to blame.
                done = replace(acks_word, unit=acks_word.unit * acks.acks_per_chunk)
                device_outputs.append(OutputCounters(acks.name, done, done, done))
            uid = ttnn.cluster.get_chip_unique_id_from_fabric_node_id(mesh_device, int(node.mesh_id), int(node.chip_id))
            devices.append(
                DeviceCounters(
                    uid=uid,
                    r=r,
                    c=c,
                    inputs=[_input(s.name, s.service.get_input_counters(coord)) for s in inputs],
                    outputs=device_outputs,
                    acks=acks_word,
                )
            )
    return devices


class StreamHangDetector:
    """Handle on one stage's watcher process: python <the file that defines the watcher's class> <probe.json>."""

    def __init__(self, watcher: StreamHangWatcher) -> None:
        fd, probe_path = tempfile.mkstemp(prefix="stream_hang_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(asdict(watcher), f, separators=(",", ":"))
        self._probe_path = probe_path
        # Its file ends by running the class's main() on the hand-off.
        self._proc = subprocess.Popen([sys.executable, inspect.getfile(type(watcher)), probe_path])

    def stop(self) -> None:
        """Stop the watcher before anything it reads goes away. Safe to call twice."""
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        # In case the watcher died before loading it.
        Path(self._probe_path).unlink(missing_ok=True)
