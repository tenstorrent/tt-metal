# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pin this process to its own share of the host's CPUs, then run the given command.

tt-run starts every rank through this file. tt-metal pins each device worker thread to one CPU the process may run
on, taking the lowest ones first, so ranks that share a host and a CPU mask pin their workers onto the same cores.
Giving each local rank a disjoint set of whole physical cores keeps them apart.

    python rank_cpu_binding.py [--exclude CPULIST] -- PROGRAM [ARGS...]

The rank takes its share only when Open MPI reports at least two ranks on this host and every rank gets at least
MIN_CORES_PER_RANK cores. ``--exclude`` takes CPUs and their hyperthread siblings away from the process, for example
cores a storage client keeps busy, whether or not the ranks are split.

This file runs before PROGRAM in every rank, so it imports only the standard library.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

PREFIX = "[tt-run]"
MIN_CORES_PER_RANK = 2
LOCAL_RANK_ENV = "OMPI_COMM_WORLD_LOCAL_RANK"
LOCAL_SIZE_ENV = "OMPI_COMM_WORLD_LOCAL_SIZE"
# With this tt-metal option on, DeviceManager binds threads across every host CPU, ignoring the process's CPU mask.
NUMA_AFFINITY_ENV = "TT_METAL_NUMA_BASED_AFFINITY"
EXEC_FAILED_EXIT_CODE = 127


def parse_cpu_list(text: str) -> set[int]:
    """Parse a Linux CPU list such as ``"1,2"`` or ``"0-3,8"``."""
    cpus: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        first, sep, last = part.partition("-")
        if not first.isdigit() or (sep and not last.isdigit()):
            raise ValueError(f"invalid CPU list {text!r}: expected numbers and ranges such as 1,2 or 0-3")
        low, high = int(first), int(last) if sep else int(first)
        if high < low:
            raise ValueError(f"invalid CPU range {part!r} in {text!r}")
        cpus.update(range(low, high + 1))
    return cpus


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Format CPUs as a compact list such as ``"0-3,8"``."""
    ordered = sorted(cpus)
    runs = []
    for cpu in ordered:
        if runs and cpu == runs[-1][1] + 1:
            runs[-1][1] = cpu
        else:
            runs.append([cpu, cpu])
    return ",".join(str(low) if low == high else f"{low}-{high}" for low, high in runs)


def thread_siblings(cpu: int) -> set[int]:
    """The CPUs that share a physical core with ``cpu``, itself included."""
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    try:
        return parse_cpu_list(path.read_text())
    except (OSError, ValueError):
        return {cpu}


def numa_node(cpu: int) -> int:
    """The NUMA node ``cpu`` belongs to, or 0 when the kernel does not say."""
    try:
        for entry in os.listdir(f"/sys/devices/system/cpu/cpu{cpu}"):
            if entry.startswith("node") and entry[4:].isdigit():
                return int(entry[4:])
    except OSError:
        pass
    return 0


def physical_cores(cpus: Iterable[int], siblings: Callable[[int], set[int]] = thread_siblings) -> list[list[int]]:
    """Group ``cpus`` into physical cores, each as its sorted CPUs, ordered by lowest CPU."""
    cores: dict[int, list[int]] = {}
    remaining = set(cpus)
    for cpu in sorted(remaining):
        if not any(cpu in core for core in cores.values()):
            core = sorted(siblings(cpu) & remaining)
            cores[core[0]] = core
    return [cores[first] for first in sorted(cores)]


def usable_cores(
    allowed: Iterable[int], exclude: Iterable[int] = (), siblings: Callable[[int], set[int]] = thread_siblings
) -> list[list[int]]:
    """Physical cores in ``allowed`` with no CPU in ``exclude`` or sharing a core with one."""
    excluded = set()
    for cpu in exclude:
        excluded |= siblings(cpu) | {cpu}
    return [core for core in physical_cores(allowed, siblings) if not excluded.intersection(core)]


def cpus_for_local_rank(
    allowed: Iterable[int],
    local_rank: int,
    local_size: int,
    exclude: Iterable[int] = (),
    siblings: Callable[[int], set[int]] = thread_siblings,
    node_of: Callable[[int], int] = numa_node,
) -> Optional[set[int]]:
    """The CPUs local rank ``local_rank`` of ``local_size`` should run on, or None when the ranks are not split.

    Each NUMA node's usable cores are split into ``local_size`` contiguous runs, so every rank keeps CPUs on every
    node; tt-metal pins a device's workers to CPUs on the device's node. A node's leftover cores go to the next ranks
    in turn, keeping every rank's total within one core of the others. Every rank on a host computes the same split
    from the same inputs, so the shares never overlap.
    """
    if local_size < 2 or not 0 <= local_rank < local_size:
        return None
    cores = usable_cores(allowed, exclude, siblings)
    if len(cores) // local_size < MIN_CORES_PER_RANK:
        return None
    by_node: dict[int, list[list[int]]] = {}
    for core in cores:
        by_node.setdefault(node_of(core[0]), []).append(core)
    share: list[list[int]] = []
    first_extra = 0
    for node in sorted(by_node):
        node_cores = by_node[node]
        per_rank, extra = divmod(len(node_cores), local_size)
        counts = [per_rank + (1 if (rank - first_extra) % local_size < extra else 0) for rank in range(local_size)]
        start = sum(counts[:local_rank])
        share += node_cores[start : start + counts[local_rank]]
        first_extra = (first_extra + extra) % local_size
    return {cpu for core in share for cpu in core}


def _split_args(argv: list[str]) -> tuple[set[int], list[str]]:
    exclude: set[int] = set()
    args = list(argv)
    while args and args[0] != "--":
        if args[0] == "--exclude" and len(args) >= 2:
            exclude |= parse_cpu_list(args[1])
            args = args[2:]
        else:
            raise SystemExit(f"{PREFIX} rank_cpu_binding: unexpected argument {args[0]!r}")
    if len(args) < 2:
        raise SystemExit(f"{PREFIX} rank_cpu_binding: expected -- PROGRAM [ARGS...]")
    return exclude, args[1:]


def _say(message: str) -> None:
    print(f"{PREFIX} {message}", file=sys.stderr, flush=True)


def main(argv: list[str]) -> None:
    exclude, program = _split_args(argv)
    try:
        local_rank = int(os.environ[LOCAL_RANK_ENV])
        local_size = int(os.environ[LOCAL_SIZE_ENV])
    except (KeyError, ValueError):
        local_rank, local_size = 0, 1
    allowed = os.sched_getaffinity(0)
    if os.environ.get(NUMA_AFFINITY_ENV, "").startswith("1"):
        # tt-metal would move threads back onto other ranks' and excluded CPUs, so a mask set here would not hold.
        if exclude:
            raise SystemExit(f"{PREFIX} --exclude cannot be honored with {NUMA_AFFINITY_ENV}=1; unset one of them")
        if local_size >= 2 and local_rank == 0:
            _say(f"{NUMA_AFFINITY_ENV}=1 binds threads across every host CPU; not splitting ranks")
        return _exec(program)
    cpus = cpus_for_local_rank(allowed, local_rank, local_size, exclude, thread_siblings, numa_node)
    if cpus is not None:
        os.sched_setaffinity(0, cpus)
        _say(f"local rank {local_rank} of {local_size} on CPUs {format_cpu_list(cpus)}")
    else:
        if local_size >= 2 and local_rank == 0:
            _say(
                f"{local_size} ranks share this host but its usable CPUs give fewer than {MIN_CORES_PER_RANK} "
                "cores each; not splitting them"
            )
        if exclude:
            kept = {cpu for core in usable_cores(allowed, exclude, thread_siblings) for cpu in core}
            if not kept:
                raise SystemExit(
                    f"{PREFIX} excluding {format_cpu_list(exclude)} leaves none of CPUs {format_cpu_list(allowed)}; "
                    "narrow --rank-cpu-exclude"
                )
            os.sched_setaffinity(0, kept)
            _say(f"local rank {local_rank} on CPUs {format_cpu_list(kept)}, without {format_cpu_list(exclude)}")
    _exec(program)


def _exec(program: list[str]) -> None:
    try:
        os.execvp(program[0], program)
    except OSError as e:
        _say(f"cannot run {program[0]!r}: {e.strerror}")
        sys.exit(EXEC_FAILED_EXIT_CODE)


if __name__ == "__main__":
    main(sys.argv[1:])
