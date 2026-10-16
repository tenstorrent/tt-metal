# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""sum_ops — y = a + b[0] + ... + b[P - 1] for bf16 TILE DRAM tensors in one generic_op over the whole grid: a [R, W]
and b [P R, W] (P blocks of R rows, e.g. the K3 block's routed output plus the shared expert's P part outputs). One
pass over DRAM and one bf16 rounding, instead of ttnn.sum over the parts then ttnn.add (two passes, two roundings)."""

import ttnn

_READ = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t P = get_compile_time_arg_val(0), NT = get_compile_time_arg_val(1), BT = get_compile_time_arg_val(2);
constexpr auto a_args = TensorAccessorArgs<3>();
constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0), b_addr = get_arg_val<uint32_t>(1);
    const uint32_t nu = get_arg_val<uint32_t>(2);
    const auto a = TensorAccessor(a_args, a_addr, 2048);
    const auto b = TensorAccessor(b_args, b_addr, 2048);
    for (uint32_t u = 0; u < nu; ++u) {  // (a unit: BT tiles of a, then BT of each part)
        const uint32_t t = get_arg_val<uint32_t>(3 + u), n = NT - t < BT ? NT - t : BT;
        cb_reserve_back(0, (P + 1) * BT);
        const uint32_t l1 = get_write_ptr(0);
        if (__PROBE__ != 1) for (uint32_t i = 0; i < n; ++i) noc_async_read(a.get_noc_addr(t + i), l1 + i * 2048, 2048);
        if (__PROBE__ != 1) for (uint32_t p = 0; p < P; ++p) {
            for (uint32_t i = 0; i < n; ++i) noc_async_read(b.get_noc_addr(p * NT + t + i), l1 + ((p + 1) * BT + i) * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(0, (P + 1) * BT);
    }
}
"""

_WRITE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t BT = get_compile_time_arg_val(0), NT = get_compile_time_arg_val(1);
constexpr auto y_args = TensorAccessorArgs<2>();
void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0), nu = get_arg_val<uint32_t>(1);
    const auto y = TensorAccessor(y_args, y_addr, 2048);
    for (uint32_t u = 0; u < nu; ++u) {
        const uint32_t t = get_arg_val<uint32_t>(2 + u), n = NT - t < BT ? NT - t : BT;
        cb_wait_front(16, BT);
        const uint32_t l1 = get_read_ptr(16);
        for (uint32_t i = 0; i < n; ++i) noc_async_write(l1 + i * 2048, y.get_noc_addr(t + i), 2048);
        noc_async_writes_flushed();
        cb_pop_front(16, BT);
    }
    noc_async_write_barrier();
}
"""

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
constexpr uint32_t P = get_compile_time_arg_val(0), BT = get_compile_time_arg_val(1), NT = get_compile_time_arg_val(2);
void kernel_main() {
    const uint32_t nu = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 0, 16);
    for (uint32_t u = 0; u < nu; ++u) {
        const uint32_t t = get_arg_val<uint32_t>(1 + u), n = NT - t < BT ? NT - t : BT;
        cb_wait_front(0, (P + 1) * BT);
        tile_regs_acquire();
        if (__PROBE__ != 2) {
        add_init(0, 0);
        for (uint32_t i = 0; i < n; ++i) add_tiles(0, 0, i, BT + i, i);
        if constexpr (P > 1) {
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0);
            for (uint32_t p = 2; p <= P; ++p) {
                for (uint32_t i = 0; i < n; ++i) add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0, p * BT + i, i);
            }
        }
        }
        tile_regs_commit();
        cb_pop_front(0, (P + 1) * BT);
        cb_reserve_back(16, BT);
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) pack_tile(i, 16, i);
        tile_regs_release();
        cb_push_back(16, BT);
    }
}
"""


def sum_parts_prepare(mesh_device, *, P, bt=8, dist="rr"):
    """run(a [R, W], b [P R, W]) -> y [R, W] (bf16 TILE, DRAM interleaved)."""
    grid = mesh_device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    state = {}

    def build(a, b, y):
        NT = int(a.shape[-2]) * int(a.shape[-1]) // 1024
        starts = list(range(0, NT, bt))  # (units of bt tiles, dealt round robin: concurrent units are neighbours)
        units = {i: starts[i :: len(cores)] for i in range(len(cores))}
        if dist == "contig":
            per = -(-len(starts) // len(cores))
            units = {i: starts[i * per : (i + 1) * per] for i in range(len(cores))}
        used = [i for i in range(len(cores)) if units[i]]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(cores[i], cores[i]) for i in used])
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i in used:
            c, us = cores[i], units[i]
            rd[c.x][c.y] = [int(a.buffer_address()), int(b.buffer_address()), len(us)] + us
            wr[c.x][c.y] = [int(y.buffer_address()), len(us)] + us
            cp[c.x][c.y] = [len(us)] + us
        a_ct = list(ttnn.TensorAccessorArgs(a).get_compile_time_args())
        b_ct = list(ttnn.TensorAccessorArgs(b).get_compile_time_args())
        y_ct = list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
        program = ttnn.ProgramDescriptor()
        program.cbs = [
            ttnn.CBDescriptor(total_size=2 * (P + 1) * bt * 2048, core_ranges=crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048)]),
            ttnn.CBDescriptor(total_size=2 * bt * 2048, core_ranges=crs,
                              format_descriptors=[ttnn.CBFormatDescriptor(16, ttnn.bfloat16, 2048)]),
        ]
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc)  # noqa: E731
        import os

        pr = {"noread": "1", "nocompute": "2"}.get(os.environ.get("SUM_PROBE", ""), "0")
        swap = os.environ.get("SUM_NOC", "swap") == "swap"  # (DRAM reads on NOC0: 52 vs 75 us on bh-lb-09, 12 x 10)
        program.kernels = [
            ttnn.KernelDescriptor(kernel_source=_READ.replace("__PROBE__", pr), source_type=st, core_ranges=crs, compile_time_args=[P, NT, bt] + a_ct + b_ct,
                                  runtime_args=rd, config=dm("RISCV_1", ttnn.NOC.NOC_0 if swap else ttnn.NOC.RISCV_1_default)),
            ttnn.KernelDescriptor(kernel_source=_WRITE, source_type=st, core_ranges=crs, compile_time_args=[bt, NT] + y_ct,
                                  runtime_args=wr, config=dm("RISCV_0", ttnn.NOC.NOC_1 if swap else ttnn.NOC.RISCV_0_default)),
            ttnn.KernelDescriptor(kernel_source=_COMPUTE.replace("__PROBE__", pr), source_type=st, core_ranges=crs, compile_time_args=[P, bt, NT],
                                  runtime_args=cp, config=ttnn.ComputeConfigDescriptor()),
        ]
        R, C = tuple(mesh_device.shape)
        mesh_desc = ttnn.MeshProgramDescriptor()
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = program
        return mesh_desc

    def run(a, b):
        y = ttnn.allocate_tensor_on_device(ttnn.Shape(list(a.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device,
                                           ttnn.DRAM_MEMORY_CONFIG)
        k = tuple(int(t.buffer_address()) for t in (a, b, y))
        if state.get("key") != k:
            state["key"], state["desc"] = k, build(a, b, y)
        ttnn.generic_op([a, b, y], state["desc"])
        return y

    return run
