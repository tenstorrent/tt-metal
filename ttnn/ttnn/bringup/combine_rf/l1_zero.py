# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""l1_zero — zero runs of L1 on chosen cores, on the device (one generic_op on every chip of the mesh).

For per-call scratch that a kernel expects zero at the start (counters, flags): a freshly allocated L1 tensor holds
whatever the previous op left there. Runs are (address, bytes, stride, count): count blocks of `bytes` (16-byte
multiples) every `stride` bytes. A run's first 2 KB are zeroed by stores, the rest by local NoC copies of that."""

import ttnn

_ZERO = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0);
    uint32_t zsrc = 0, zlen = 0;  // a zeroed block to copy from
    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t addr = get_arg_val<uint32_t>(1 + 4 * r), bytes = get_arg_val<uint32_t>(2 + 4 * r);
        const uint32_t stride = get_arg_val<uint32_t>(3 + 4 * r), count = get_arg_val<uint32_t>(4 + 4 * r);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t b = addr + i * stride;
            uint32_t done = 0;
            if (zlen < 2048 && bytes > zlen) {  // (stores: the first block, and small runs before there is one)
                const uint32_t m = bytes < 2048 ? bytes : 2048;
                volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b);
                for (uint32_t w = 0; w < m / 4; ++w) p[w] = 0;
                done = m;
                if (m > zlen) {
                    zsrc = b;
                    zlen = m;
                }
            }
            while (done < bytes) {
                const uint32_t c = bytes - done < zlen ? bytes - done : zlen;
                noc_async_write(zsrc, get_noc_addr(b + done), c);
                done += c;
            }
            noc_async_write_barrier();  // (the copies' source may be overwritten by a later run: none overlap)
        }
    }
}
"""


def l1_zero_program(mesh_device, runs_by_core):
    """runs_by_core[(logical x, y)] = [(address, bytes, stride, count)]: the same on every chip. Returns the
    MeshProgramDescriptor (run it with ttnn.generic_op(io_tensors, desc))."""
    cores = sorted(runs_by_core)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    rt = ttnn.RuntimeArgs()
    for x, y in cores:
        runs = runs_by_core[(x, y)]
        for a, b, _, _ in runs:
            assert a % 16 == 0 and b % 16 == 0, "16-byte aligned runs"
        rt[x][y] = [len(runs)] + [v for r in runs for v in r]
    program = ttnn.ProgramDescriptor()
    program.kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_ZERO,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=crs,
            compile_time_args=[],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
            ),
        )
    ]
    R, C = tuple(mesh_device.shape)
    desc = ttnn.MeshProgramDescriptor()
    desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = program
    ttnn.freeze_program_hashes(desc)  # (hash stored once: not re-hashed per call)
    return desc


_ZERO_SIGNAL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
// zero my runs (each (NoC x, y, address, bytes, stride, count): my own first, by stores then local NoC copies; then
// other cores' from my first zeroed block), then count in on every port of the chip (its zero gate)
void kernel_main() {
    size_t a = 0;
    const uint32_t n = get_arg_val<uint32_t>(a++), ctr = get_arg_val<uint32_t>(a++), np = get_arg_val<uint32_t>(a++);
    const uint32_t ports = a, runs = a + 2 * np;
    const uint32_t my = get_noc_addr(0) >> NOC_ADDR_LOCAL_BITS;
    uint32_t zsrc = 0, zlen = 0;
    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t x = get_arg_val<uint32_t>(runs + 6 * r), y = get_arg_val<uint32_t>(runs + 6 * r + 1);
        const uint32_t addr = get_arg_val<uint32_t>(runs + 6 * r + 2), bytes = get_arg_val<uint32_t>(runs + 6 * r + 3);
        const uint32_t stride = get_arg_val<uint32_t>(runs + 6 * r + 4), count = get_arg_val<uint32_t>(runs + 6 * r + 5);
        const bool local = (get_noc_addr(x, y, 0) >> NOC_ADDR_LOCAL_BITS) == my;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t b = addr + i * stride;
            uint32_t done = 0;
            if (local && zlen < 2048 && bytes > zlen) {  // (stores: the first block)
                const uint32_t m = bytes < 2048 ? bytes : 2048;
                volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b);
                for (uint32_t w = 0; w < m / 4; ++w) p[w] = 0;
                done = m;
                if (m > zlen) {
                    zsrc = b;
                    zlen = m;
                }
            }
            while (done < bytes) {
                const uint32_t c = bytes - done < zlen ? bytes - done : zlen;
                noc_async_write(zsrc, get_noc_addr(x, y, b + done), c);
                done += c;
            }
            noc_async_write_barrier();  // (the copies' source may be overwritten by a later run: none overlap)
        }
    }
    for (uint32_t i = 0; i < np; ++i)
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(ports + 2 * i), get_arg_val<uint32_t>(ports + 2 * i + 1), ctr), 1);
    noc_async_atomic_barrier();
}
"""


def l1_zero_kernels(mesh_device, runs_by_core, busy, ports_of, ctr_addr):
    """l1_zero as kernels of another program (its fn(coord) -> [KernelDescriptor] for `extra` / a zero gate):
    runs_by_core as l1_zero_program's (the same on every chip); the cores in `busy` (taken by that program's own
    kernels) get theirs zeroed by another core of the list (NoC writes from its first zeroed block - it zeroes its own
    first, so it must have one of 2 KB); ports_of(coord): the cores (logical) each kernel counts in on (ctr_addr: +1 a
    kernel) once its runs are zeroed. Returns (fn, number of kernels a chip)."""
    virt = mesh_device.worker_core_from_logical_core
    own = {c: list(r) for c, r in runs_by_core.items() if c not in busy}
    helpers = sorted((c for c, r in own.items() if any(b >= 2048 for _, b, _, _ in r)), key=lambda c: len(own[c]))
    assert helpers or not (set(runs_by_core) & set(busy)), "l1_zero_kernels: no core to zero the busy ones"
    remote = {c: [] for c in own}
    for i, c in enumerate(sorted(set(runs_by_core) & set(busy))):
        remote[helpers[i % len(helpers)]].append(c)
    cores = sorted(own)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])

    def fn(coord):
        pv = [virt(ttnn.CoreCoord(*p_)) for p_ in ports_of(coord)]
        rt = ttnn.RuntimeArgs()
        for x, y in cores:
            v = virt(ttnn.CoreCoord(x, y))
            runs = [(v.x, v.y) + tuple(r) for r in own[(x, y)]]  # (my own first: the source of the rest)
            for t in remote[(x, y)]:
                tv = virt(ttnn.CoreCoord(*t))
                runs += [(tv.x, tv.y) + tuple(r) for r in runs_by_core[t]]
            for _, _, a, b, _, _ in runs:
                assert a % 16 == 0 and b % 16 == 0, "16-byte aligned runs"
            rt[x][y] = (
                [len(runs), ctr_addr, len(pv)] + [w for q in pv for w in (q.x, q.y)] + [w for r in runs for w in r]
            )
        return [
            ttnn.KernelDescriptor(
                kernel_source=_ZERO_SIGNAL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=[],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        ]

    return fn, len(cores)
