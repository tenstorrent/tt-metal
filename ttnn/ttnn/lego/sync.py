# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host side of the sync blocks (kernels/sync/*): stage hand-off between core groups of one device program."""

from __future__ import annotations

import ttnn

from .compose import Block, LegoProgram

KDIR = "ttnn/operations/experimental/lego/kernels"


def _xy(device, core):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
    return [c.x, c.y]


def place_barrier(prog: LegoProgram, device, cores: list, *, risc: str, label="bar"):
    """All `cores` meet: each one's earlier writes (on `risc`) are visible to all after it."""
    arrive = prog.sem(cores, 0)
    release = prog.sem(cores, 0)
    leader = cores[0]
    members = [v for c in cores for v in _xy(device, c)]
    for is_leader, sel in ((True, [leader]), (False, cores[1:])):
        if sel:
            prog.place(
                sel,
                Block(
                    cls="lego::sync::Barrier",
                    header=f"{KDIR}/sync/barrier.hpp",
                    risc=risc,
                    ct=dict(arrive_sem=arrive, release_sem=release, num_members=len(cores), is_leader=is_leader),
                    rt=lambda c, il=is_leader: _xy(device, leader) + (members if il else []),
                    label=label,
                ),
            )


def place_handoff(
    prog: LegoProgram, device, producers: list, producer_risc: str, consumers: list, consumer_risc: str, label="hand"
):
    """Consumers wait until every producer's writes (issued on producer_risc before this point) have landed."""
    sem = prog.sem(list(dict.fromkeys(producers + consumers)), 0)
    targets = [v for c in consumers for v in _xy(device, c)]
    prog.place(
        producers,
        Block(
            cls="lego::sync::Signal",
            header=f"{KDIR}/sync/signal.hpp",
            risc=producer_risc,
            ct=dict(sem=sem),
            rt=lambda c: [len(consumers)] + targets,
            label=f"{label}_sig",
        ),
    )
    prog.place(
        consumers,
        Block(
            cls="lego::sync::Wait",
            header=f"{KDIR}/sync/wait.hpp",
            risc=consumer_risc,
            ct=dict(sem=sem, count=len(producers)),
            label=f"{label}_wait",
        ),
    )
    return sem


def place_handoff_multi(
    prog: LegoProgram,
    device,
    producers: list,
    consumers: list,
    consumer_risc: str,
    label="hand",
    sem: int | None = None,
):
    """Like place_handoff with several producer groups: producers = [(cores, risc), ...]. Consumers wait until every
    producer (core, risc) has signalled. Returns (semaphore id, count)."""
    all_cores = list(dict.fromkeys([c for cs, _ in producers for c in cs] + consumers))
    sem = prog.sem(all_cores, 0) if sem is None else sem
    targets = [v for c in consumers for v in _xy(device, c)]
    count = 0
    for cores, risc in producers:
        prog.place(
            cores,
            Block(
                cls="lego::sync::Signal",
                header=f"{KDIR}/sync/signal.hpp",
                risc=risc,
                ct=dict(sem=sem),
                rt=lambda c: [len(consumers)] + targets,
                label=f"{label}_sig",
            ),
        )
        count += len(cores)
    if consumer_risc is not None:
        prog.place(
            consumers,
            Block(
                cls="lego::sync::Wait",
                header=f"{KDIR}/sync/wait.hpp",
                risc=consumer_risc,
                ct=dict(sem=sem, count=count),
                label=f"{label}_wait",
            ),
        )
    return sem, count


def place_local_handoff(prog: LegoProgram, device, cores: list, producer_risc: str, consumer_risc: str, label="lh"):
    """On each core, consumer_risc waits until producer_risc on the same core has passed this point (its writes
    landed). E.g. one RISC's next block reuses L1 that another RISC's previous block was still using."""
    sem = prog.sem(cores, 0)
    prog.place(
        cores,
        Block(
            cls="lego::sync::Signal",
            header=f"{KDIR}/sync/signal.hpp",
            risc=producer_risc,
            ct=dict(sem=sem),
            rt=lambda c: [1] + _xy(device, c),
            label=f"{label}_sig",
        ),
    )
    prog.place(
        cores,
        Block(
            cls="lego::sync::Wait",
            header=f"{KDIR}/sync/wait.hpp",
            risc=consumer_risc,
            ct=dict(sem=sem, count=1),
            label=f"{label}_wait",
        ),
    )
    return sem
