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
    return desc
