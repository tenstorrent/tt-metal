// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm — worker compute. Per block (block_rows x core_w_tiles):
//   sumsq_block            : eltwise_chain x*x accumulated in DEST per tile row -> cb_sq_acc (1 tile per row);
//                            reduce<SUM, REDUCE_ROW, Accurate> -> post-op transpose_dest -> cb_partial
//                            (row 0 = the 32 per-row partial sums: the wire record layout).
//   gamma_prescale_block   : (HAS_GAMMA) x * gamma (Row bcast) -> cb_xg; runs while the exchange is in flight.
//   combine_finalize_block : reduce<SUM, REDUCE_COL, Accurate> over the G*S_w source rows -> post-op
//                            transpose_dest, x 1/(G*W_local), + eps, rsqrt -> cb_rstd (col 0 = rstd per row).
//   normalize_block        : (xg | x) * rstd (Col bcast) -> cb_output_tiles.
//
// Post-op bodies are raw SFPU / transpose calls on a DEST tile inside reduce (no chain element can act on the
// reduce's DEST tile; transpose_dest has no chain element). The reduce helper programs the packer's reduce edge
// mask (keeps only the reduced column/row); after the transpose the valid data is in the other orientation, so
// the post-op clears that mask before the pack, and re-arms the datacopy MOP the helper's next tile expects.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

using namespace compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sq_acc = get_compile_time_arg_val(3);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(4);
    constexpr uint32_t cb_xg = get_compile_time_arg_val(5);
    constexpr uint32_t cb_stats_combine = get_compile_time_arg_val(6);
    constexpr uint32_t cb_rstd = get_compile_time_arg_val(7);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(8);
    constexpr bool has_gamma = get_compile_time_arg_val(9) != 0;

    uint32_t a = 0;
    const uint32_t core_rows = get_arg_val<uint32_t>(a++);
    const uint32_t block_rows = get_arg_val<uint32_t>(a++);
    const uint32_t cw = get_arg_val<uint32_t>(a++);
    const uint32_t n_src_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(a++);  // fp32 bits of 1 / (G * W_local)
    const uint32_t eps_bits = get_arg_val<uint32_t>(a++);    // fp32 bits of epsilon

    if (core_rows == 0) {
        return;  // idle core of the worker rectangle
    }

    compute_kernel_hw_startup(cb_x_resident, cb_scaler, cb_output_tiles);

    // Row 0 <-> col 0 in DEST, then let the packer write the whole tile again.
    auto sumsq_post = [](uint32_t dst) {
        transpose_dest_init<DST_ACCUM_MODE, true>();
        transpose_dest<DST_ACCUM_MODE, true>(dst);
        PACK((llk_pack_reduce_mask_clear()));
        copy_init(cb_sq_acc);
    };
    auto finalize_post = [inv_w_bits, eps_bits](uint32_t dst) {
        transpose_dest_init<DST_ACCUM_MODE, true>();
        transpose_dest<DST_ACCUM_MODE, true>(dst);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, inv_w_bits);
        add_unary_tile(dst, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst);
        PACK((llk_pack_reduce_mask_clear()));
        copy_init(cb_stats_combine);
    };

    constexpr uint32_t cb_norm_src = has_gamma ? cb_xg : cb_x_resident;

    for (uint32_t r0 = 0; r0 < core_rows; r0 += block_rows) {
        const uint32_t rows = (core_rows - r0) < block_rows ? (core_rows - r0) : block_rows;

        // ---- sumsq_block: x^2 accumulated across the W chunk in DEST, one packed tile per row ----
        eltwise_chain(
            IterationShape::grid(rows, cw),
            BinaryFpu<
                BinaryFpuOp::Mul,
                input(cb_x_resident, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block),
                input(
                    input(cb_x_resident, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block),
                    BroadcastDim::None),
                Dst::D0,
                DestAccumulation::PerRow>{},
            PackTile<output(
                cb_sq_acc,
                ReservePolicy::PerOuter,
                PushPolicy::PerOuter,
                DataFormatReconfig::Enabled,
                TileAddressing::Direct,
                DestAccumulation::PerRow)>{});
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_sq_acc,
            cb_scaler,
            cb_partial,
            ReduceInputPolicy::BulkWaitBulkPop,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Accurate>(
            ReduceInputBlockShape::of(rows, 1), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, sumsq_post);

        // ---- gamma_prescale_block: in the exchange stall (x*rstd*gamma == (x*gamma)*rstd) ----
        if constexpr (has_gamma) {
            eltwise_chain(
                IterationShape::grid(rows, cw),
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(cb_x_resident, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                    input(
                        input(cb_gamma_tiles, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                        BroadcastDim::Row)>{},
                PackTile<output(cb_xg)>{});
        }

        // ---- combine_finalize_block: sources are rows of n_src_tiles combine tiles per tile row ----
        // Tiles are laid out row-major (source tile k of tile row j at k * rows + j), so the helper waits for the
        // whole block upfront and indexes it; the pop is ours (WaitUpfrontNoPop hands it to the caller).
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            cb_stats_combine,
            cb_scaler,
            cb_rstd,
            ReduceInputPolicy::WaitUpfrontNoPop,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Accurate>(
            ReduceInputBlockShape::of(n_src_tiles, rows),
            ReduceInputMemoryLayout::contiguous(),
            NoAccumulation{},
            finalize_post);
        cb_pop_front(cb_stats_combine, n_src_tiles * rows);

        // ---- normalize_block: (xg | x) * rstd, rstd broadcast along each tile row ----
        eltwise_chain(
            IterationShape::grid(rows, cw),
            BinaryFpu<
                BinaryFpuOp::Mul,
                input(cb_norm_src),
                input(
                    input(cb_rstd, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col), BroadcastDim::Col)>{},
            PackTile<output(cb_output_tiles)>{});
    }
}
