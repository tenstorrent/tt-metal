# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""glu_act - the clamped SiLU GLU in one op: h [M, P] = silu(min(g, L)) * clamp(u, -L, L) from gu [M, 2 P] (bf16 TILE,
per row tile the P gate tiles then the P up tiles: one matmul's output), L = 10 (the SFPU's ClampedSiluGluConfigDsV4,
DeepSeek V4 / GLM-5.3 swiglu_limit). Units (row tile, CW output tiles) over the whole grid; DRAM interleaved in / out.
Replaces slice, slice, minimum, silu, clamp, multiply (six programs, ~23 us each at [320, 1024])."""

import ttnn

_READ = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t CW = get_compile_time_arg_val(0), P = get_compile_time_arg_val(1);
constexpr auto src_args = TensorAccessorArgs<2>();
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1);
    const auto src = TensorAccessor(src_args, src_addr, 2048);
    for (uint32_t u = 0; u < n; ++u) {
        const uint32_t w = get_arg_val<uint32_t>(2 + u), tr = w >> 16, c = w & 0xFFFF;
        cb_reserve_back(0, 2 * CW);
        const uint32_t l1 = get_write_ptr(0);
        for (uint32_t i = 0; i < CW; ++i) {  // (gate tile, up tile) pairs
            noc_async_read(src.get_noc_addr(tr * 2 * P + c * CW + i), l1 + 2 * i * 2048, 2048);
            noc_async_read(src.get_noc_addr(tr * 2 * P + P + c * CW + i), l1 + (2 * i + 1) * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(0, 2 * CW);
    }
}
"""

_WRITE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t CW = get_compile_time_arg_val(0), P = get_compile_time_arg_val(1);
constexpr auto dst_args = TensorAccessorArgs<2>();
void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1);
    const auto dst = TensorAccessor(dst_args, dst_addr, 2048);
    for (uint32_t u = 0; u < n; ++u) {
        const uint32_t w = get_arg_val<uint32_t>(2 + u), tr = w >> 16, c = w & 0xFFFF;
        cb_wait_front(16, CW);
        const uint32_t l1 = get_read_ptr(16);
        for (uint32_t i = 0; i < CW; ++i) noc_async_write(l1 + i * 2048, dst.get_noc_addr(tr * P + c * CW + i), 2048);
        noc_async_write_barrier();
        cb_pop_front(16, CW);
    }
}
"""

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_clamped_silu_glu.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif
constexpr uint32_t CW = get_compile_time_arg_val(0);
void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0);
    init_sfpu(0, 16);
    copy_tile_init(0);
    MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::clamped_silu_glu_init)));
    for (uint32_t u = 0; u < n; ++u) {
        cb_wait_front(0, 2 * CW);
        cb_reserve_back(16, CW);
        for (uint32_t i = 0; i < CW; ++i) {
            tile_regs_acquire();
            copy_tile(0, 2 * i, 0);
            copy_tile(0, 2 * i + 1, 1);
            MATH((SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_clamped_silu_glu,
                (DST_ACCUM_MODE, 8, sfpu::ClampedSiluGluConfigDsV4),
                0,
                1,
                0,
                VectorMode::RC)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, 16);
            tile_regs_release();
        }
        cb_push_back(16, CW);
        cb_pop_front(0, 2 * CW);
    }
}
"""


def glu_act_prepare(mesh_device, *, cw=4):
    """run(gu [M, 2 P] bf16 TILE) -> h [M, P] bf16 TILE (DRAM interleaved); P % cw == 0."""
    grid = mesh_device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(  # noqa: E731
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    state = {}

    def build(gu, h, M, P):
        units = [(tr << 16) | c for tr in range(M // 32) for c in range(P // cw)]
        per = {i: units[i :: len(cores)] for i in range(len(cores))}
        used = [i for i in range(len(cores)) if per[i]]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(cores[i], cores[i]) for i in used])
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i in used:
            c_ = cores[i]
            rd[c_.x][c_.y] = [int(gu.buffer_address()), len(per[i])] + per[i]
            wr[c_.x][c_.y] = [int(h.buffer_address()), len(per[i])] + per[i]
            cp[c_.x][c_.y] = [len(per[i])]
        program = ttnn.ProgramDescriptor()
        program.cbs = [
            ttnn.CBDescriptor(
                total_size=2 * 2 * cw * 2048,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048)],
            ),
            ttnn.CBDescriptor(
                total_size=2 * cw * 2048,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(16, ttnn.bfloat16, 2048)],
            ),
        ]
        program.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_READ,
                source_type=st,
                core_ranges=crs,
                compile_time_args=[cw, P] + list(ttnn.TensorAccessorArgs(gu).get_compile_time_args()),
                runtime_args=rd,
                config=dm("RISCV_1", ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_WRITE,
                source_type=st,
                core_ranges=crs,
                compile_time_args=[cw, P] + list(ttnn.TensorAccessorArgs(h).get_compile_time_args()),
                runtime_args=wr,
                config=dm("RISCV_0", ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_COMPUTE,
                source_type=st,
                core_ranges=crs,
                compile_time_args=[cw],
                runtime_args=cp,
                config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
            ),
        ]
        R, C = tuple(mesh_device.shape)
        md = ttnn.MeshProgramDescriptor()
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = program
        ttnn.freeze_program_hashes(md)
        return md

    def run(gu):
        M, P2 = int(gu.shape[-2]), int(gu.shape[-1])
        P = P2 // 64
        assert M % 32 == 0 and P2 % 64 == 0 and P % cw == 0, (M, P2, cw)
        h = ttnn.allocate_tensor_on_device(
            ttnn.Shape([int(gu.shape[i]) for i in range(len(gu.shape) - 1)] + [P2 // 2]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        k = (int(gu.buffer_address()), int(h.buffer_address()), M, P)
        if k not in state:  # (descriptors by buffer addresses: a few per shape)
            if len(state) > 64:
                state.clear()
            state[k] = build(gu, h, M, P)
        ttnn.generic_op([gu, h], state[k])
        return h

    return run
