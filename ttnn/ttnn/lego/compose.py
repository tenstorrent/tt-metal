# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lego composer: build one fused program per device out of kernel blocks.

A *block* is a logical unit of an existing op (a matmul's in0 reader, a reduce-scatter's writer, ...), written as a
C++ class template in ttnn/cpp/ttnn/operations/experimental/lego/kernels with init() / run() / teardown(). The host
places block instances on cores and RISCs; the composer then

  * allocates circular buffers and semaphores per core (one index space per core, so blocks that share a core never
    collide, and two blocks share a CB only when the caller passes the same index to both),
  * groups cores whose RISC carries the same block list into one kernel and generates its source: a struct of
    constexpr values per block (its compile-time values plus the offsets of its runtime-arg slices) and a
    kernel_main() that calls every init(), then every run(), then every teardown(), in placement order,
  * concatenates each core's per-block runtime args and each kernel's positional compile-time args (tensor
    accessors), and returns a ttnn.ProgramDescriptor.

Kernels are therefore made only of blocks: the composer writes no kernel logic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Iterable

import ttnn

Core = tuple[int, int]

RISCS = ("dm0", "dm1", "compute")
NUM_CBS = 64
NUM_SEMAPHORES = 16


def core_range_set(cores: Iterable[Core]) -> ttnn.CoreRangeSet:
    """Cores -> CoreRangeSet, merging each row's consecutive x runs."""
    by_row: dict[int, list[int]] = {}
    for x, y in sorted(set(cores), key=lambda c: (c[1], c[0])):
        by_row.setdefault(y, []).append(x)
    ranges = []
    for y, xs in by_row.items():
        start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(start, y), ttnn.CoreCoord(prev, y)))
            if x is not None:
                start = prev = x
    return ttnn.CoreRangeSet(ranges)


def cores_of(spec) -> list[Core]:
    """Accept (x, y), a list of them, a CoreRange or a CoreRangeSet."""
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], int):
        return [spec]
    if isinstance(spec, ttnn.CoreRange):
        spec = ttnn.CoreRangeSet([spec])
    if isinstance(spec, ttnn.CoreRangeSet):
        return [(c.x, c.y) for c in ttnn.corerange_to_cores(spec)]
    return [tuple(c) for c in spec]


@dataclass
class Block:
    """One block instance.

    cls      C++ template, e.g. "lego::mm::In0" (instantiated with the generated args struct)
    header   path of its header relative to ttnn/cpp, e.g. "ttnn/operations/experimental/lego/kernels/mm/in0.hpp"
    risc     "dm0" (RISCV_0 / BRISC), "dm1" (RISCV_1 / NCRISC) or "compute"
    ct       compile-time values: name -> int | bool  (CB indices and semaphore ids go here too)
    rt       per-core runtime args: callable core -> list[int], or dict core -> list[int]
    crt      common runtime args
    tensors  name -> ttnn.Tensor: appends its TensorAccessorArgs to the kernel's positional CT args and sets
             cta_<name> to their offset
    defines  preprocessor defines this block needs (must agree with the other blocks of the kernel)
    compute  compute-config requirements (math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode)
    label    name of the instance in generated code (default b<i>)
    """

    cls: str
    header: str
    risc: str
    ct: dict = field(default_factory=dict)
    rt: Callable[[Core], list[int]] | dict | None = None
    crt: list[int] = field(default_factory=list)
    tensors: dict = field(default_factory=dict)
    defines: dict = field(default_factory=dict)
    compute: dict = field(default_factory=dict)
    label: str | None = None

    def rt_for(self, core: Core) -> list[int]:
        if self.rt is None:
            return []
        if callable(self.rt):
            return [int(v) for v in self.rt(core)]
        return [int(v) for v in self.rt.get(core, [])]

    def cta_values(self) -> list[tuple[str, list[int]]]:
        out = []
        for name, t in self.tensors.items():
            args = t if isinstance(t, list) else list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
            out.append((name, args))
        return out

    def signature(self):
        return (
            self.cls,
            self.header,
            tuple(
                sorted((k, int(v) if not isinstance(v, bool) else v) for k, v in self.ct.items() if k != "hw_startup")
            ),
            tuple(sorted(self.defines.items())),
            tuple((n, tuple(a)) for n, a in self.cta_values()),
            tuple(sorted(self.compute.items(), key=lambda kv: kv[0])),
            len(self.crt),
            tuple(self.crt),
        )


@dataclass
class Arena:
    """An L1 region per core (one shard of a HEIGHT_SHARDED L1 tensor) that the host carves into circular buffers at
    chosen offsets. CBs whose lifetimes do not overlap may share bytes: the host, not the CB allocator, decides how
    long each byte range holds what."""

    tensor: object
    cores: list
    size: int

    @staticmethod
    def allocate(mesh, cores, size: int) -> "Arena":
        size = -(-size // 128) * 128
        rows = size // 128
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_range_set(cores), [rows, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        t = ttnn.allocate_tensor_on_device(
            ttnn.Shape([rows * len(cores), 32]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, mem
        )
        return Arena(t, list(cores), size)


@dataclass
class _Placement:
    block: Block
    cores: list[Core]


class LegoProgram:
    """One device's fused program."""

    def __init__(self, dm_noc=None):
        self._placements: dict[str, list[_Placement]] = {r: [] for r in RISCS}
        self._cbs: list[tuple[list[Core], int, ttnn.CBDescriptor]] = []
        self._cb_used: dict[Core, set[int]] = {}
        self._sems: list[tuple[list[Core], int, int]] = []
        self._sem_used: dict[Core, set[int]] = {}
        # NOC per data-movement RISC (blocks on a RISC share its NOC); set_noc overrides it per core
        self.dm_noc = dm_noc or {"dm0": ttnn.NOC.NOC_0, "dm1": ttnn.NOC.NOC_1}
        self._noc: dict[tuple, object] = {}
        self.compute_defaults = {}

    # ------------------------------------------------------------- resources

    def cb(
        self,
        cores,
        num_pages: int,
        page_size: int,
        data_format,
        *,
        tile: ttnn.TileDescriptor | None = None,
        index: int | None = None,
        buffer=None,
        arena: tuple | None = None,
    ) -> int:
        """Allocate a circular buffer on `cores`; returns its index (the lowest one free on every core).
        arena = (Arena, byte offset): place it in that arena instead of letting the program allocate it."""
        cores = cores_of(cores)
        if index is None:
            index = next(i for i in range(NUM_CBS) if all(i not in self._cb_used.get(c, set()) for c in cores))
        for c in cores:
            used = self._cb_used.setdefault(c, set())
            assert index not in used, f"CB {index} already used on core {c}"
            used.add(index)
        fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)
        if tile is not None:
            fmt.tile = tile
        if arena is not None:
            ar, off = arena
            assert off % 32 == 0 and off + num_pages * page_size <= ar.size, (off, num_pages * page_size, ar.size)
            assert set(cores) <= set(ar.cores), "CB cores must be inside the arena"
            desc = ttnn.cb_descriptor_from_sharded_tensor(
                index,
                ar.tensor,
                address_offset=off,
                total_size=num_pages * page_size,
                core_ranges=core_range_set(cores),
            )
            desc.format_descriptors = [fmt]
        else:
            desc = ttnn.CBDescriptor(
                total_size=num_pages * page_size, core_ranges=core_range_set(cores), format_descriptors=[fmt]
            )
            if buffer is not None:
                desc.set_buffer_from_tensor(buffer)
        self._cbs.append((cores, index, desc))
        return index

    def sem(self, cores, initial: int = 0) -> int:
        """Allocate a (program-local) semaphore on `cores`; returns its id."""
        cores = cores_of(cores)
        sid = next(i for i in range(NUM_SEMAPHORES) if all(i not in self._sem_used.get(c, set()) for c in cores))
        for c in cores:
            self._sem_used.setdefault(c, set()).add(sid)
        self._sems.append((cores, sid, initial))
        return sid

    def set_noc(self, cores, risc: str, noc):
        """Use `noc` for `risc` on `cores` (every block placed there shares it)."""
        for c in cores_of(cores):
            self._noc[(c, risc)] = noc

    # ------------------------------------------------------------- placement

    def place(self, cores, block: Block):
        assert block.risc in RISCS, block.risc
        self._placements[block.risc].append(_Placement(block, cores_of(cores)))
        return block

    # ------------------------------------------------------------- build

    def _kernels_for(self, risc: str) -> list[ttnn.KernelDescriptor]:
        # Per core: the ordered blocks on this RISC.
        per_core: dict[Core, list[Block]] = {}
        for p in self._placements[risc]:
            for c in p.cores:
                per_core.setdefault(c, []).append(p.block)
        # Group cores by the signature of their block list.
        groups: dict[tuple, list[Core]] = {}
        blocks_of: dict[tuple, list[Block]] = {}
        for c, blocks in per_core.items():
            key = (str(self._noc.get((c, risc), self.dm_noc.get(risc))),) + tuple(b.signature() for b in blocks)
            groups.setdefault(key, []).append(c)
            blocks_of.setdefault(key, blocks)
        kernels = []
        for key, cores in groups.items():
            kernels.append(self._kernel(risc, blocks_of[key], cores, per_core))
        return kernels

    def _kernel(self, risc: str, blocks: list[Block], cores: list[Core], per_core) -> ttnn.KernelDescriptor:
        # Runtime args come from each core's own block instances (same signature as `blocks`, possibly different
        # closures); each block's slice starts at the same offset on every core of the group.
        rt_len = [max(len(per_core[c][i].rt_for(c)) for c in cores) for i in range(len(blocks))]
        rt_base = [sum(rt_len[:i]) for i in range(len(blocks))]
        crt_base = [sum(len(b.crt) for b in blocks[:i]) for i in range(len(blocks))]
        cta, cta_base = [], []
        for b in blocks:
            offs = {}
            for name, args in b.cta_values():
                offs[name] = len(cta)
                cta.extend(args)
            cta_base.append(offs)

        import os

        defines: dict[str, str] = {"LEGO_ZONES": "1"} if os.environ.get("LEGO_ZONES") == "1" else {}
        compute: dict = dict(self.compute_defaults)
        for b in blocks:
            for k, v in b.defines.items():
                assert defines.get(k, v) == v, f"define {k} conflicts between blocks on one kernel"
                defines[k] = v
            for k, v in b.compute.items():
                assert compute.get(k, v) == v, f"compute config {k} conflicts between blocks on one kernel"
                compute[k] = v

        labels = [b.label or f"b{i}" for i, b in enumerate(blocks)]
        src = self._source(blocks, labels, rt_base, crt_base, cta_base)

        runtime_args = ttnn.RuntimeArgs()
        for c in cores:
            args = []
            for b, n in zip(per_core[c], rt_len):
                a = b.rt_for(c)
                args.extend(a + [0] * (n - len(a)))
            runtime_args[c[0]][c[1]] = args
        common = [v for b in blocks for v in b.crt]

        if risc == "compute":
            cfg = ttnn.ComputeConfigDescriptor()
            for k, v in compute.items():
                setattr(cfg, k, v)
        else:
            cfg = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0 if risc == "dm0" else ttnn.DataMovementProcessor.RISCV_1,
                noc=self._noc.get((cores[0], risc), self.dm_noc[risc]),
            )
        return ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core_range_set(cores),
            compile_time_args=cta,
            defines=list(defines.items()),
            runtime_args=runtime_args,
            common_runtime_args=common,
            config=cfg,
        )

    @staticmethod
    def _source(blocks, labels, rt_base, crt_base, cta_base) -> str:
        # compute_kernel_hw_startup must be the first compute call of a kernel: blocks that take hw_startup get it
        # set on the first such block of the kernel only.
        first_hw = next((i for i, b in enumerate(blocks) if "hw_startup" in b.ct), None)
        import os

        zones = os.environ.get("LEGO_ZONES") == "1"
        lines = ["// generated by ttnn/ttnn/lego/compose.py", "#include <cstdint>"]
        if zones:
            lines.append('#include "api/debug/kernel_profiler.hpp"')
        for h in dict.fromkeys(b.header for b in blocks):
            lines.append(f'#include "{h}"')
        lines.append("namespace lego_gen {")
        for b, label, rb, cb, cta in zip(blocks, labels, rt_base, crt_base, cta_base):
            lines.append(f"struct {label}_args {{")
            lines.append(f"    static constexpr uint32_t rt_base = {rb};")
            lines.append(f"    static constexpr uint32_t crt_base = {cb};")
            for name, off in cta.items():
                lines.append(f"    static constexpr uint32_t cta_{name} = {off};")
            for k, v in b.ct.items():
                if k == "hw_startup":
                    v = blocks.index(b) == first_hw
                if isinstance(v, bool):
                    lines.append(f"    static constexpr bool {k} = {'true' if v else 'false'};")
                else:
                    lines.append(f"    static constexpr uint32_t {k} = {int(v)}u;")
            lines.append("};")
        lines.append("}  // namespace lego_gen")
        lines.append("void kernel_main() {")
        for b, label in zip(blocks, labels):
            lines.append(f"    {b.cls}<lego_gen::{label}_args> {label};")
        for phase in ("init", "run", "teardown"):
            for label in labels:
                if zones and phase == "run":
                    lines.append(f'    {{ DeviceZoneScopedN("{label}"); {label}.run(); }}')
                else:
                    lines.append(f"    {label}.{phase}();")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def descriptor(self) -> ttnn.ProgramDescriptor:
        p = ttnn.ProgramDescriptor()
        p.kernels = [k for r in RISCS for k in self._kernels_for(r)]
        p.cbs = [d for _, _, d in self._cbs]
        p.semaphores = [
            ttnn.SemaphoreDescriptor(id=sid, core_ranges=core_range_set(cores), initial_value=init)
            for cores, sid, init in self._sems
        ]
        return p

    def sources(self) -> list[str]:
        """Generated kernel sources (for inspection)."""
        return [k.kernel_source for k in self.descriptor().kernels]


def mesh_descriptor(programs: dict[tuple[int, int], ttnn.ProgramDescriptor]) -> ttnn.MeshProgramDescriptor:
    desc = ttnn.MeshProgramDescriptor()
    for (r, c), p in programs.items():
        coord = ttnn.MeshCoordinate(r, c)
        desc[ttnn.MeshCoordinateRange(coord, coord)] = p
    return desc


def noc_xy(device, core: Core, noc: int = 0) -> tuple[int, int]:
    """Physical NOC coordinates of a logical worker core."""
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
    return c.x, c.y
