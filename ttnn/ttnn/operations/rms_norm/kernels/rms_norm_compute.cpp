// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm compute. Per row block (Rb tile-rows), all through kernel_lib helpers:
//   pass 1, per Sb-tile segment:  [tilize (RM)] -> square -> cb_sq -> reduce<SUM,ROW> (Accumulate across
//           segments) -> cb_stat
//   inv_rms (once per block):     cb_stat <- rsqrt(cb_stat * (1/W) + eps), in place (fp32 SFPU)
//   pass 2, per segment:          [tilize again if not resident] x * inv_rms (Col bcast) -> (cb_xn) ->
//           [* gamma (Row bcast)] -> cb_out_tiles -> [untilize (RM) -> cb_out_rm]
//
// Deviations from the design (recorded in the commit): no cb_accum (cb_stat is both the Accumulate
// accumulator and the reduce output); the gamma multiply is a second chain through cb_xn because
// DestReuseBinary cannot express a Row broadcast (design-sanctioned fallback). All CBs use the default
// unpack mode (FPU consumers read through SrcA/SrcB, so UnpackToDestFp32 would be incorrect for them).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

// The inv_rms chain runs on a reduce(ROW) output: only column 0 of each tile is meaningful, and that column
// lives in faces 0 and 2. VectorMode::C runs the SFPU on exactly those two faces (half the SFPU work of the
// default all-faces ops, which dominates the per-tile-row cost on short rows).
struct AddEpsCol : ckl::UnaryOp<AddEpsCol, ckl::Dst::D0> {
    uint32_t param0;
    constexpr explicit AddEpsCol(uint32_t p) noexcept : param0(p) {}
    static ALWI void init() { binop_with_scalar_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { add_unary_tile(slot_offset, param0, VectorMode::C); }
};
struct RsqrtCol : ckl::UnaryOp<RsqrtCol, ckl::Dst::D0> {
    static ALWI void init() { rsqrt_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { rsqrt_tile(slot_offset, VectorMode::C); }
};

void kernel_main() {
    constexpr bool IS_RM = get_compile_time_arg_val(0) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(1) != 0;
    constexpr bool RESIDENT = get_compile_time_arg_val(2) != 0;
    constexpr bool GAMMA_RESIDENT = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t Sb = get_compile_time_arg_val(5);
    constexpr uint32_t Rb = get_compile_time_arg_val(6);
    constexpr uint32_t num_seg = get_compile_time_arg_val(7);
    constexpr uint32_t cb_x_rm = get_compile_time_arg_val(8);
    constexpr uint32_t cb_x = get_compile_time_arg_val(9);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(10);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(11);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(12);
    constexpr uint32_t cb_stat = get_compile_time_arg_val(13);
    constexpr uint32_t cb_xn = get_compile_time_arg_val(14);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(15);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(16);
    constexpr uint32_t N_W = get_compile_time_arg_val(17);  // W-split row-group width; 1 = off
    constexpr uint32_t cb_psend = get_compile_time_arg_val(18);
    constexpr uint32_t cb_part = get_compile_time_arg_val(19);
    constexpr bool PRESCALED = get_compile_time_arg_val(20) != 0;  // 1/W already folded into the reduce scaler
    constexpr bool SPLIT = N_W > 1;

    const uint32_t rows_core = get_arg_val<uint32_t>(0);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_x, cb_scaler, cb_out_tiles);

    DataflowBuffer dfb_x_rm(cb_x_rm);
    DataflowBuffer dfb_x(cb_x);
    DataflowBuffer dfb_gamma(cb_gamma);
    DataflowBuffer dfb_sq(cb_sq);
    DataflowBuffer dfb_stat(cb_stat);
    DataflowBuffer dfb_part(cb_part);
    DataflowBuffer dfb_psend(cb_psend);
    DataflowBuffer dfb_xn(cb_xn);
    DataflowBuffer dfb_out_tiles(cb_out_tiles);
    DataflowBuffer dfb_out_rm(cb_out_rm);

    // A operand of the x-tile consumers: resident -> caller-managed window addressed at a runtime base;
    // streamed -> per-tile front-relative.
    constexpr auto x_in = RESIDENT ? ckl::input(
                                         cb_x,
                                         ckl::WaitPolicy::None,
                                         ckl::PopPolicy::None,
                                         ckl::InputTileMapping::Block,
                                         ckl::DataFormatReconfig::Enabled,
                                         ckl::TileAddressing::Offset)
                                   : ckl::input(cb_x);
    constexpr auto stat_bcast = ckl::input(
        cb_stat,
        ckl::BroadcastDim::Col,
        ckl::WaitPolicy::Upfront,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Col);
    constexpr auto gamma_bcast = GAMMA_RESIDENT ? ckl::input(
                                                      cb_gamma,
                                                      ckl::BroadcastDim::Row,
                                                      ckl::WaitPolicy::None,
                                                      ckl::PopPolicy::None,
                                                      ckl::InputTileMapping::Row,
                                                      ckl::DataFormatReconfig::Enabled,
                                                      ckl::TileAddressing::Offset)
                                                : ckl::input(
                                                      cb_gamma,
                                                      ckl::BroadcastDim::Row,
                                                      ckl::WaitPolicy::Upfront,
                                                      ckl::PopPolicy::AtEnd,
                                                      ckl::InputTileMapping::Row);
    constexpr uint32_t cb_scale_out = HAS_GAMMA ? cb_xn : cb_out_tiles;

    for (uint32_t b = 0; b < rows_core; b += Rb) {
        const uint32_t rb = (rows_core - b) < Rb ? (rows_core - b) : Rb;
        const auto grid = ckl::IterationShape::grid(rb, Sb);
        const uint32_t seg_tiles = rb * Sb;

        // ---------------- pass 1: Σx² ----------------
        for (uint32_t seg = 0; seg < num_seg; ++seg) {
            if constexpr (IS_RM) {
                ckl::tilize<Sb, cb_x_rm, cb_x>(dfb_x_rm, dfb_x, rb, TILE_HEIGHT * rb);
            }
            if constexpr (RESIDENT) {
                const uint32_t base = seg * seg_tiles;
                dfb_x.wait_front(base + seg_tiles);
                ckl::eltwise_chain(
                    grid,
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_in, x_in>{dfb_x, dfb_x, base, base},
                    ckl::PackTile<ckl::output(cb_sq)>{dfb_sq});
            } else {
                ckl::square<ckl::input(cb_x), ckl::output(cb_sq)>(grid, dfb_x, dfb_sq);
            }
            ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, cb_sq, cb_scaler, cb_stat>(
                ckl::ReduceInputBlockShape::of(rb, Sb, 1),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(cb_stat, seg));
        }

        // ---------------- W-split combine: cb_stat[rr] = sum_j partial[rr][j] (L1-accumulating pack) ----
        if constexpr (SPLIT) {
            // Hand this core's finished partial (Accumulate keeps cb_stat in flux until the last segment)
            // to the writer, which exchanges it with the row group; the sum is rebuilt into cb_stat.
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(rb),
                ckl::CopyTile<ckl::input(cb_stat)>{dfb_stat},
                ckl::PackTile<ckl::output(cb_psend)>{dfb_psend});
            dfb_part.wait_front(Rb * N_W);
            for (uint32_t rr = 0; rr < rb; ++rr) {
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(N_W),
                    ckl::CopyTile<ckl::input(cb_part)>{dfb_part},
                    ckl::PackTile<ckl::output(
                        cb_stat,
                        ckl::ReservePolicy::OneUpfront,
                        ckl::PushPolicy::OneAtEnd,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Direct,
                        ckl::DestAccumulation::Disabled,
                        ckl::L1Accumulation::Enabled)>{dfb_stat});
            }
            if (rb < Rb) {
                dfb_part.pop_front((Rb - rb) * N_W);  // unused rows of the block (ring stays in step)
            }
        }

        // ---------------- inv_rms (in place on cb_stat) ----------------
        if constexpr (PRESCALED) {
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(rb),
                ckl::CopyTile<ckl::input(cb_stat)>{dfb_stat},
                AddEpsCol{eps_bits},
                RsqrtCol{},
                ckl::PackTile<ckl::output(cb_stat)>{dfb_stat});
        } else {
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(rb),
                ckl::CopyTile<ckl::input(cb_stat)>{dfb_stat},
                ckl::MulUnary<>{inv_w_bits},
                AddEpsCol{eps_bits},
                RsqrtCol{},
                ckl::PackTile<ckl::output(cb_stat)>{dfb_stat});
        }

        // ---------------- pass 2: normalize ----------------
        if constexpr (HAS_GAMMA && GAMMA_RESIDENT) {
            if (b == 0) {
                dfb_gamma.wait_front(Wt);  // first use; the reader loads gamma while pass 1 runs
            }
        }
        for (uint32_t seg = 0; seg < num_seg; ++seg) {
            if constexpr (IS_RM && !RESIDENT) {
                ckl::tilize<Sb, cb_x_rm, cb_x>(dfb_x_rm, dfb_x, rb, TILE_HEIGHT * rb);
            }
            if constexpr (RESIDENT) {
                const uint32_t base = seg * seg_tiles;
                ckl::eltwise_chain(
                    grid,
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_in, stat_bcast>{dfb_x, dfb_stat, base},
                    ckl::PackTile<ckl::output(cb_scale_out)>{HAS_GAMMA ? dfb_xn : dfb_out_tiles});
            } else {
                ckl::eltwise_chain(
                    grid,
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_in, stat_bcast>{dfb_x, dfb_stat},
                    ckl::PackTile<ckl::output(cb_scale_out)>{HAS_GAMMA ? dfb_xn : dfb_out_tiles});
            }
            if constexpr (HAS_GAMMA) {
                if constexpr (GAMMA_RESIDENT) {
                    ckl::eltwise_chain(
                        grid,
                        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, ckl::input(cb_xn), gamma_bcast>{
                            dfb_xn, dfb_gamma, 0u, seg * Sb},
                        ckl::PackTile<ckl::output(cb_out_tiles)>{dfb_out_tiles});
                } else {
                    ckl::eltwise_chain(
                        grid,
                        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, ckl::input(cb_xn), gamma_bcast>{dfb_xn, dfb_gamma},
                        ckl::PackTile<ckl::output(cb_out_tiles)>{dfb_out_tiles});
                }
            }
            if constexpr (IS_RM) {
                ckl::untilize<Sb, cb_out_tiles, cb_out_rm>(dfb_out_tiles, dfb_out_rm, rb);
            }
        }

        if constexpr (RESIDENT) {
            dfb_x.pop_front(rb * Wt);
        }
        dfb_stat.pop_front(rb);
    }
}
