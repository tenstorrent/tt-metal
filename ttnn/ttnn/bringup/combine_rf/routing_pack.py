# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""routing_pack — the top-k ids (uint16 TILE [S, K]) and weights (bf16 TILE [S, K]) -> one row-major uint16 [S, 2 K]
tensor, each token's K ids then its K weights' bits (what the routing gather and the planner read), in one generic_op:
ttnn's two untilizes, the bitcast and the concat (which transposes) were four ops and ~0.3 ms of host per call.

A core takes whole tile rows: it reads the 32-row tile of ids and of weights (K <= 16: a row's K values are the first
2 K bytes of its face row, faces 0 / 2), packs the 32 rows in L1 and writes them as pages.
"""

import ttnn

_PACK = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t K = get_compile_time_arg_val(0), RB = 4 * K;  // (a packed row: K ids, K weights, 2 bytes each)
constexpr auto ids_args = TensorAccessorArgs<1>();
constexpr auto w_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
void kernel_main() {
    const uint32_t ids_addr = get_arg_val<uint32_t>(0), w_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2), tr0 = get_arg_val<uint32_t>(3), ntr = get_arg_val<uint32_t>(4);
    const auto ids = TensorAccessor(ids_args, ids_addr, 2048);
    const auto w = TensorAccessor(w_args, w_addr, 2048);
    const auto out = TensorAccessor(out_args, out_addr, RB);
    const uint32_t l1 = get_write_ptr(0), stage = l1 + 4096;
    for (uint32_t tr = tr0; tr < tr0 + ntr; ++tr) {
        noc_async_read(ids.get_noc_addr(tr), l1, 2048);
        noc_async_read(w.get_noc_addr(tr), l1 + 2048, 2048);
        noc_async_read_barrier();
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t off = (r < 16 ? 0 : 1024) + (r % 16) * 32;  // (face 0 / 2, face row r % 16)
            volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage + r * RB);
            volatile tt_l1_ptr uint16_t* a = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1 + off);
            volatile tt_l1_ptr uint16_t* b = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1 + 2048 + off);
            for (uint32_t k = 0; k < K; ++k) {
                d[k] = a[k];
                d[K + k] = b[k];
            }
        }
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_write(stage + r * RB, out.get_noc_addr(tr * 32 + r), RB);
        }
        noc_async_write_barrier();
    }
}
"""


def routing_pack_prepare(mesh_device, *, max_cores=10):
    """run(ids uint16 TILE [S, K], wts bf16 TILE [S, K], out uint16 ROW_MAJOR [S, 2 K]) -> out; S % 32 == 0, K <= 16.
    The same program on every chip; its descriptors are kept per buffer-address set (a few per layer in a model)."""
    grid = mesh_device.compute_with_storage_grid_size()
    cache = {}

    def build(ids, wts, out):
        S, K = int(ids.shape[-2]), int(ids.shape[-1])
        assert S % 32 == 0 and K <= 16 and list(out.shape)[-2:] == [S, 2 * K], (S, K, list(out.shape))
        assert ids.layout == ttnn.TILE_LAYOUT and wts.layout == ttnn.TILE_LAYOUT and out.layout == ttnn.ROW_MAJOR_LAYOUT
        nt = S // 32
        n = min(nt, max_cores)
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        rt = ttnn.RuntimeArgs()
        a = [int(t.buffer_address()) for t in (ids, wts, out)]
        for i, c in enumerate(cores):
            t0, t1 = i * nt // n, (i + 1) * nt // n
            rt[c.x][c.y] = a + [t0, t1 - t0]
        ct = [K]
        for t in (ids, wts, out):
            ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        program = ttnn.ProgramDescriptor()
        program.cbs = [
            ttnn.CBDescriptor(
                total_size=4096 + 32 * 4 * K,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint16, 4096 + 32 * 4 * K)],
            )
        ]
        program.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_PACK,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=ct,
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

    def run(ids, wts, out):
        k = tuple(int(t.buffer_address()) for t in (ids, wts, out)) + (tuple(ids.shape),)
        desc = cache.get(k)
        if desc is None:
            if len(cache) >= 64:
                cache.pop(next(iter(cache)))
            desc = cache[k] = build(ids, wts, out)
        ttnn.generic_op([ids, wts, out], desc)
        return out

    return run
