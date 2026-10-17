# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""layout_ops — bf16 tilize / untilize of a [rows, W] DRAM-interleaved tensor, parallel over (row tile, chunk of CW
tile columns) units on the whole grid (one generic_op). ttnn.tilize / untilize split a [320, W] tensor only over its
10 row tiles (K3's latent up input: 53 us for 2.3 MB; the shared expert's x: 49 us untilize at W 7168).

  tilize    reader: a unit's 32 rows x CW * 32 columns (512-byte stick pieces at CW 8) into the in CB; compute:
            compute_kernel_lib::tilize; writer: CW tiles to their pages tr * Wt + c * CW + i
  untilize  the reverse (reader: CW tiles; compute: compute_kernel_lib::untilize; writer: 32 stick pieces)
"""

import ttnn

_READ_ROWS = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t CW = get_compile_time_arg_val(0), WB = get_compile_time_arg_val(1), P = get_compile_time_arg_val(2);
constexpr uint32_t PB = WB / P;  // (a row as P pages of PB bytes)
constexpr auto src_args = TensorAccessorArgs<3>();
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1);
    const auto src = TensorAccessor(src_args, src_addr, PB);
    for (uint32_t u = 0; u < n; ++u) {
        const uint32_t w = get_arg_val<uint32_t>(2 + u), tr = w >> 16, c = w & 0xFFFF;
        cb_reserve_back(0, CW);
        const uint32_t l1 = get_write_ptr(0);
        const uint32_t b = c * CW * 64;
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_read(src.get_noc_addr((tr * 32 + r) * P + b / PB) + b % PB, l1 + r * CW * 64, CW * 64);
        }
        noc_async_read_barrier();
        cb_push_back(0, CW);
    }
}
"""

_WRITE_TILES = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t CW = get_compile_time_arg_val(0), WT = get_compile_time_arg_val(1);
constexpr auto dst_args = TensorAccessorArgs<2>();
void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1);
    const auto dst = TensorAccessor(dst_args, dst_addr, 2048);
    for (uint32_t u = 0; u < n; ++u) {
        const uint32_t w = get_arg_val<uint32_t>(2 + u), tr = w >> 16, c = w & 0xFFFF;
        cb_wait_front(16, CW);
        const uint32_t l1 = get_read_ptr(16);
        for (uint32_t i = 0; i < CW; ++i) noc_async_write(l1 + i * 2048, dst.get_noc_addr(tr * WT + c * CW + i), 2048);
        noc_async_write_barrier();
        cb_pop_front(16, CW);
    }
}
"""

_READ_TILES = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t CW = get_compile_time_arg_val(0), WT = get_compile_time_arg_val(1);
constexpr auto src_args = TensorAccessorArgs<2>();
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1);
    const auto src = TensorAccessor(src_args, src_addr, 2048);
    for (uint32_t u = 0; u < n; ++u) {
        const uint32_t w = get_arg_val<uint32_t>(2 + u), tr = w >> 16, c = w & 0xFFFF;
        cb_reserve_back(0, CW);
        const uint32_t l1 = get_write_ptr(0);
        for (uint32_t i = 0; i < CW; ++i) noc_async_read(src.get_noc_addr(tr * WT + c * CW + i), l1 + i * 2048, 2048);
        noc_async_read_barrier();
        cb_push_back(0, CW);
    }
}
"""

_WRITE_ROWS = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t CW = get_compile_time_arg_val(0), WB = get_compile_time_arg_val(1), P = get_compile_time_arg_val(2);
constexpr uint32_t PB = WB / P;  // (a row as P pages of PB bytes)
constexpr auto dst_args = TensorAccessorArgs<3>();
void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1), row0 = get_arg_val<uint32_t>(2);
    const auto dst = TensorAccessor(dst_args, dst_addr, PB);
    for (uint32_t u = 0; u < n; ++u) {
        const uint32_t w = get_arg_val<uint32_t>(3 + u), tr = w >> 16, c = w & 0xFFFF;
        cb_wait_front(16, CW);
        const uint32_t l1 = get_read_ptr(16);
        const uint32_t b = c * CW * 64;
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_write(l1 + r * CW * 64, dst.get_noc_addr((row0 + tr * 32 + r) * P + b / PB) + b % PB, CW * 64);
        }
        noc_async_write_barrier();
        cb_pop_front(16, CW);
    }
}
"""

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/__KIND___helpers.hpp"
constexpr uint32_t CW = get_compile_time_arg_val(0);
void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 16);
    compute_kernel_lib::__KIND__<CW, 0, 16,
        compute_kernel_lib::__KIND___config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::__KIND___config::WaitMode::WaitBlock,
        compute_kernel_lib::__KIND___config::ReconfigureRegisterDatatypeMode::NoReconfigure>(n);
}
"""


def layout_op_prepare(mesh_device, kind, *, cw=8, pages_per_row=1, out_memory_config=None, row0_of=None):
    """kind "tilize": run(src [rows, W] bf16 ROW_MAJOR) -> [rows, W] bf16 TILE; "untilize": the reverse. DRAM
    interleaved in and out; rows % 32 == 0, W % (32 * cw) == 0. pages_per_row P: the row-major side stores each row as
    P pages ([rows * P, W / P], e.g. P 2 at W 7168: pages that fit a fabric packet)."""
    assert kind in ("tilize", "untilize")
    grid = mesh_device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc)  # noqa: E731
    comp = _COMPUTE.replace("__KIND__", kind)
    if kind == "tilize":
        comp = comp.replace('#include "api/compute/tilize.h"\n', '#include "api/compute/tilize.h"\n')
    else:
        comp = comp.replace('#include "api/compute/tilize.h"\n', "")
    state = {}

    P = pages_per_row
    out_mc = out_memory_config or ttnn.DRAM_MEMORY_CONFIG  # (e.g. a block-sharded L1 output: the accessor handles it)

    def build(src, dst, rows, W):  # (untilize, row0_of (mesh row, column) -> the first dst row: one program per chip)
        Wt, nt = W // 32, rows // 32
        assert (W * 2 // P) % (cw * 64) == 0, "a CW-tile chunk must not straddle pages"
        assert rows % 32 == 0 and Wt % cw == 0
        units = [(tr << 16) | c for tr in range(nt) for c in range(Wt // cw)]
        per = {i: units[i :: len(cores)] for i in range(len(cores))}
        used = [i for i in range(len(cores)) if per[i]]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(cores[i], cores[i]) for i in used])
        def args(row0):
            rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i in used:
                c_ = cores[i]
                rd_rt[c_.x][c_.y] = [int(src.buffer_address()), len(per[i])] + per[i]
                wr_rt[c_.x][c_.y] = [int(dst.buffer_address()), len(per[i])] + ([row0] if kind == "untilize" else []) + per[i]
                cp_rt[c_.x][c_.y] = [len(per[i])]
            return rd_rt, wr_rt, cp_rt
        s_ct = list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
        d_ct = list(ttnn.TensorAccessorArgs(dst).get_compile_time_args())
        if kind == "tilize":
            rd_src, rd_ct, wr_src, wr_ct = _READ_ROWS, [cw, W * 2, P] + s_ct, _WRITE_TILES, [cw, Wt] + d_ct
        else:
            rd_src, rd_ct, wr_src, wr_ct = _READ_TILES, [cw, Wt] + s_ct, _WRITE_ROWS, [cw, W * 2, P] + d_ct
        import os

        swap = os.environ.get("LAYOUT_NOC", "swap") == "swap"  # (reads on NOC0, writes on NOC1)

        def program_for(row0):
            rd_rt, wr_rt, cp_rt = args(row0)
            program = ttnn.ProgramDescriptor()
            program.cbs = [
                ttnn.CBDescriptor(total_size=2 * cw * 2048, core_ranges=crs,
                                  format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048)]),
                ttnn.CBDescriptor(total_size=2 * cw * 2048, core_ranges=crs,
                                  format_descriptors=[ttnn.CBFormatDescriptor(16, ttnn.bfloat16, 2048)]),
            ]
            program.kernels = [
                ttnn.KernelDescriptor(kernel_source=rd_src, source_type=st, core_ranges=crs, compile_time_args=rd_ct,
                                      runtime_args=rd_rt, config=dm("RISCV_1", ttnn.NOC.NOC_0 if swap else ttnn.NOC.RISCV_1_default)),
                ttnn.KernelDescriptor(kernel_source=wr_src, source_type=st, core_ranges=crs, compile_time_args=wr_ct,
                                      runtime_args=wr_rt, config=dm("RISCV_0", ttnn.NOC.NOC_1 if swap else ttnn.NOC.RISCV_0_default)),
                ttnn.KernelDescriptor(kernel_source=comp, source_type=st, core_ranges=crs, compile_time_args=[cw],
                                      runtime_args=cp_rt, config=ttnn.ComputeConfigDescriptor()),
            ]
            return program

        R, C = tuple(mesh_device.shape)
        mesh_desc = ttnn.MeshProgramDescriptor()
        if row0_of is None:
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = program_for(0)
        else:
            for r in range(R):
                for c in range(C):
                    mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program_for(row0_of(r, c))
        return mesh_desc

    def run(src, dst=None):
        """dst (untilize): an existing row-major [>= row0 + rows, W] tensor (rows from row0_of), else a new one."""
        rows, W = int(src.shape[-2]), int(src.shape[-1])
        if kind == "tilize":  # (src [rows * P, W / P])
            rows, W = rows // P, W * P
            dst = ttnn.allocate_tensor_on_device(ttnn.Shape([rows, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, out_mc)
        elif dst is None:
            dst = ttnn.allocate_tensor_on_device(ttnn.Shape([rows * P, W // P]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
        k = (int(src.buffer_address()), int(dst.buffer_address()), rows, W)
        if state.get("key") != k:
            state["key"], state["desc"] = k, build(src, dst, rows, W)
        ttnn.generic_op([src, dst], state["desc"])
        return dst

    return run
