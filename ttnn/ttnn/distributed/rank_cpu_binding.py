# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pin this process to its own share of the host's CPUs, then run the given command.

tt-run starts every rank through this file. tt-metal pins each device worker thread to one CPU the process may run
on, taking the lowest ones first, so ranks that share a host and a CPU mask pin their workers onto the same cores.
Giving each local rank a disjoint set of whole physical cores keeps them apart.

    python rank_cpu_binding.py [--exclude CPULIST] -- PROGRAM [ARGS...]

The rank pins itself only when Open MPI reports at least two ranks on this host and every rank gets at least
MIN_CORES_PER_RANK cores; otherwise it runs PROGRAM unchanged. ``--exclude`` takes CPUs out of every rank's share,
together with their hyperthread siblings, for example cores a storage client keeps busy.

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


def physical_cores(cpus: Iterable[int], siblings: Callable[[int], set[int]] = thread_siblings) -> list[list[int]]:
    """Group ``cpus`` into physical cores, each as its sorted CPUs, ordered by lowest CPU."""
    cores: dict[int, list[int]] = {}
    remaining = set(cpus)
    for cpu in sorted(remaining):
        if not any(cpu in core for core in cores.values()):
            core = sorted(siblings(cpu) & remaining)
            cores[core[0]] = core
    return [cores[first] for first in sorted(cores)]


def cpus_for_local_rank(
    allowed: Iterable[int],
    local_rank: int,
    local_size: int,
    exclude: Iterable[int] = (),
    siblings: Callable[[int], set[int]] = thread_siblings,
) -> Optional[set[int]]:
    """The CPUs local rank ``local_rank`` of ``local_size`` should run on, or None to leave the process unpinned.

    Whole physical cores are split into ``local_size`` contiguous runs, the first ``len(cores) % local_size`` ranks
    taking one extra core. Every rank on a host computes the same split from the same inputs, so the runs never
    overlap. A core is left out when any of its CPUs is excluded.
    """
    if local_size < 2 or not 0 <= local_rank < local_size:
        return None
    excluded = set()
    for cpu in exclude:
        excluded |= siblings(cpu) | {cpu}
    cores = [core for core in physical_cores(allowed, siblings) if not excluded.intersection(core)]
    per_rank, extra = divmod(len(cores), local_size)
    if per_rank < MIN_CORES_PER_RANK:
        return None
    start = local_rank * per_rank + min(local_rank, extra)
    count = per_rank + (1 if local_rank < extra else 0)
    return {cpu for core in cores[start : start + count] for cpu in core}


def _split_args(argv: list[str]) -> tuple[set[int], list[str]]:
    exclude: set[int] = set()
    args = list(argv)
    while args and args[0] != "--":
        if args[0] == "--exclude" and len(args) >= 2:
            exclude = parse_cpu_list(args[1])
            args = args[2:]
        else:
            raise SystemExit(f"{PREFIX} rank_cpu_binding: unexpected argument {args[0]!r}")
    if not args or len(args) < 2:
        raise SystemExit(f"{PREFIX} rank_cpu_binding: expected -- PROGRAM [ARGS...]")
    return exclude, args[1:]


def main(argv: list[str]) -> None:
    exclude, program = _split_args(argv)
    try:
        local_rank = int(os.environ[LOCAL_RANK_ENV])
        local_size = int(os.environ[LOCAL_SIZE_ENV])
    except (KeyError, ValueError):
        local_rank, local_size = 0, 1
    allowed = os.sched_getaffinity(0)
    cpus = cpus_for_local_rank(allowed, local_rank, local_size, exclude)
    if cpus is not None:
        os.sched_setaffinity(0, cpus)
        print(
            f"{PREFIX} local rank {local_rank} of {local_size} on CPUs {format_cpu_list(cpus)}",
            file=sys.stderr,
            flush=True,
        )
    elif local_size >= 2 and local_rank == 0:
        print(
            f"{PREFIX} {local_size} ranks share this host but its CPUs {format_cpu_list(allowed)} (excluding "
            f"{format_cpu_list(exclude) or 'none'}) give fewer than {MIN_CORES_PER_RANK} cores each; not pinning",
            file=sys.stderr,
            flush=True,
        )
    os.execvp(program[0], program)


if __name__ == "__main__":
    main(sys.argv[1:])
