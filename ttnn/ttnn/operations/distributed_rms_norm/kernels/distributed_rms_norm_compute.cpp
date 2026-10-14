// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm — worker compute. Per block (block_rows x core_w_tiles):
//   sumsq_block            : eltwise_chain x*x accumulated in DEST per tile row -> cb_sq_acc (1 tile per row);
//                            reduce<SUM, REDUCE_ROW, Accurate> -> post-op transpose_dest -> cb_partial
//                            (row 0 = the 32 per-row partial sums: the wire record layout).
//   gamma_prescale_block   : (HAS_GAMMA && GAMMA_PRESCALE) x * gamma (Row bcast) -> cb_xg; runs while the exchange
//                            is in flight.
//   combine_finalize_block : reduce<SUM, REDUCE_COL, Accurate> over the G*S_w source rows -> post-op
//                            transpose_dest, x 1/(G*W_local), + eps, rsqrt -> cb_rstd (col 0 = rstd per row).
//   normalize_block        : (xg | x) * rstd (Col bcast) -> cb_output_tiles; with gamma but no prescale (the host
//                            drops cb_xg when it does not fit L1) the gamma multiply is fused after the rstd multiply
//                            as a DEST-reuse Mul against row-replicated gamma tiles.
//
// Sharded x (R9/R10): cb_x_resident is backed on the resident shard (row stride cw_phys tiles), published once by
// the reader and read in place through Strided addressing, never popped. With a native sharded output,
// cb_output_tiles is the output shard: reserved once, packed in place (Strided), pushed once. The per-row stats
// CBs stream in stat_rows batches, and gamma may stream in gamma_chunk-tile chunks (both host knobs, == the
// whole block / W chunk on the interleaved path).
//   local_combine_block    : (R5 pre-combine, combiners) reduce<SUM, REDUCE_COL> over the S_w W-chunk records of
//                            each of my tile rows -> cb_lc_out; row 0 is the combined record (no transpose).
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
    constexpr bool gamma_prescale = has_gamma && get_compile_time_arg_val(10) != 0;
    constexpr bool x_sharded = get_compile_time_arg_val(11) != 0;
    constexpr bool out_native = get_compile_time_arg_val(12) != 0;
    constexpr bool local_combine = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t cb_lc = get_compile_time_arg_val(14);
    constexpr uint32_t cb_lc_out = get_compile_time_arg_val(15);
    constexpr uint32_t round_lookahead = get_compile_time_arg_val(16);  // rounds whose front half runs ahead

    uint32_t a = 0;
    const uint32_t core_rows = get_arg_val<uint32_t>(a++);
    const uint32_t block_rows = get_arg_val<uint32_t>(a++);
    const uint32_t cw = get_arg_val<uint32_t>(a++);
    const uint32_t n_src_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(a++);  // fp32 bits of 1 / (G * W_local)
    const uint32_t eps_bits = get_arg_val<uint32_t>(a++);    // fp32 bits of epsilon
    // sharded-path args (R9/R10); the interleaved path keeps its Phase 0 arg list and these defaults
    uint32_t cw_phys = cw, stat_rows = block_rows, gamma_chunk = cw, lc_rows = 0, n_lc_tiles = 0;
    if constexpr (x_sharded) {
        cw_phys = get_arg_val<uint32_t>(a++);      // row stride (tiles) of the resident x / out shard
        stat_rows = get_arg_val<uint32_t>(a++);    // tile rows per stats batch (knob)
        gamma_chunk = get_arg_val<uint32_t>(a++);  // gamma tiles per chunk (knob; == cw: one chunk)
        lc_rows = get_arg_val<uint32_t>(a++);      // tile rows I pre-combine (0: not a combiner)
        n_lc_tiles = get_arg_val<uint32_t>(a++);   // ceil(S_w / 32)
    }

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

    constexpr uint32_t cb_norm_src = gamma_prescale ? cb_xg : cb_x_resident;
    // x block (sharded): the reader published core_rows * cw_phys resident tiles once
    constexpr auto x_in = input(
        cb_x_resident,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Strided);
    const bool gamma_chunked = gamma_chunk < cw;
    if constexpr (x_sharded) {
        cb_wait_front(cb_x_resident, core_rows * cw_phys);
    }
    if constexpr (out_native) {
        cb_reserve_back(cb_output_tiles, core_rows * cw_phys);
    }

    const uint32_t num_blocks = (core_rows + block_rows - 1) / block_rows;
    auto rows_of = [&](uint32_t b) {
        const uint32_t r0 = b * block_rows;
        return (core_rows - r0) < block_rows ? (core_rows - r0) : block_rows;
    };

    // ---- front half of round b: sumsq -> cb_partial (records), R5 local combine, gamma prescale ----
    // held: interleaved x tiles of earlier rounds still at the front of cb_x_resident (R2 without prescale: round b-1's
    // block waits there for its normalize while round b's sumsq runs ahead)
    auto front = [&](uint32_t round, uint32_t held) {
        [[maybe_unused]] const uint32_t r0 = round * block_rows;
        const uint32_t rows = rows_of(round);

        // ---- sumsq_block: x^2 accumulated across the W chunk in DEST, one packed tile per row ----
        for (uint32_t b0 = 0; b0 < rows; b0 += stat_rows) {
            const uint32_t nb = (rows - b0) < stat_rows ? (rows - b0) : stat_rows;
            constexpr auto sq_out = output(
                cb_sq_acc,
                ReservePolicy::PerOuter,
                PushPolicy::PerOuter,
                DataFormatReconfig::Enabled,
                TileAddressing::Direct,
                DestAccumulation::PerRow);
            if constexpr (x_sharded) {
                const StridedTileRange xr{(r0 + b0) * cw_phys, cw_phys};
                eltwise_chain(
                    IterationShape::grid(nb, cw),
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        x_in,
                        input(x_in, BroadcastDim::None),
                        Dst::D0,
                        DestAccumulation::PerRow>{xr, xr},
                    PackTile<sq_out>{});
            } else if (held > 0) {
                // interleaved, behind an earlier round's resident block: wait for my block, read it past `held`
                cb_wait_front(cb_x_resident, held + (b0 + nb) * cw);
                const StridedTileRange xr{held + b0 * cw, cw};
                eltwise_chain(
                    IterationShape::grid(nb, cw),
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        x_in,
                        input(x_in, BroadcastDim::None),
                        Dst::D0,
                        DestAccumulation::PerRow>{xr, xr},
                    PackTile<sq_out>{});
            } else {
                // interleaved: one block of rows (stat_rows == block_rows), tiles consumed as the reader lands them
                eltwise_chain(
                    IterationShape::grid(nb, cw),
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(cb_x_resident, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block),
                        input(
                            input(cb_x_resident, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block),
                            BroadcastDim::None),
                        Dst::D0,
                        DestAccumulation::PerRow>{},
                    PackTile<sq_out>{});
            }
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_sq_acc,
                cb_scaler,
                cb_partial,
                ReduceInputPolicy::BulkWaitBulkPop,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Accurate>(
                ReduceInputBlockShape::of(nb, 1), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, sumsq_post);
        }

        // ---- local_combine_block (R5): sum the S_w W-chunk records of each of my tile rows ----
        if constexpr (local_combine) {
            if (lc_rows > 0) {
                reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_COL,
                    cb_lc,
                    cb_scaler,
                    cb_lc_out,
                    ReduceInputPolicy::WaitUpfrontNoPop,
                    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Accurate>(
                    ReduceInputBlockShape::of(n_lc_tiles, lc_rows), ReduceInputMemoryLayout::contiguous());
                cb_pop_front(cb_lc, n_lc_tiles * lc_rows);
            }
        }

        // ---- gamma_prescale_block: in the exchange stall (x*rstd*gamma == (x*gamma)*rstd) ----
        if constexpr (gamma_prescale) {
            if constexpr (x_sharded) {
                eltwise_chain(
                    IterationShape::grid(rows, cw),
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        x_in,
                        input(
                            input(cb_gamma_tiles, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                            BroadcastDim::Row)>{StridedTileRange{r0 * cw_phys, cw_phys}},
                    PackTile<output(cb_xg)>{});
            } else {
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
        }
    };

    // ---- back half of round b: combine/finalize -> rstd, normalize -> cb_output_tiles ----
    auto back = [&](uint32_t round) {
        [[maybe_unused]] const uint32_t r0 = round * block_rows;
        const uint32_t rows = rows_of(round);
        for (uint32_t b0 = 0; b0 < rows; b0 += stat_rows) {
            const uint32_t nb = (rows - b0) < stat_rows ? (rows - b0) : stat_rows;

            // ---- combine_finalize_block: sources are rows of n_src_tiles combine tiles per tile row ----
            // Tiles are laid out row-major (source tile k of tile row j at k * nb + j), so the helper waits for the
            // whole batch upfront and indexes it; the pop is ours (WaitUpfrontNoPop hands it to the caller).
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_COL,
                cb_stats_combine,
                cb_scaler,
                cb_rstd,
                ReduceInputPolicy::WaitUpfrontNoPop,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Accurate>(
                ReduceInputBlockShape::of(n_src_tiles, nb),
                ReduceInputMemoryLayout::contiguous(),
                NoAccumulation{},
                finalize_post);
            cb_pop_front(cb_stats_combine, n_src_tiles * nb);

            // ---- normalize_block: (xg | x) * rstd, rstd broadcast along each tile row ----
            if constexpr (x_sharded) {
                constexpr auto rstd_in = input(
                    input(cb_rstd, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Col), BroadcastDim::Col);
                // normalize chunks: gamma's chunks on the fused path, else the whole W chunk
                const uint32_t chunk = (has_gamma && !gamma_prescale) ? gamma_chunk : cw;
                for (uint32_t c0 = 0; c0 < cw; c0 += chunk) {
                    const uint32_t wc = (cw - c0) < chunk ? (cw - c0) : chunk;
                    const StridedTileRange xr{(r0 + b0) * cw_phys + c0, cw_phys};
                    auto norm = [&](auto pack) {
                        if constexpr (gamma_prescale) {
                            eltwise_chain(
                                IterationShape::grid(nb, wc),
                                BinaryFpu<BinaryFpuOp::Mul, input(cb_xg), rstd_in>{},
                                pack);
                        } else if constexpr (has_gamma) {
                            eltwise_chain(
                                IterationShape::grid(nb, wc),
                                BinaryFpu<BinaryFpuOp::Mul, x_in, rstd_in>{xr},
                                DestReuseBinary<
                                    BinaryFpuOp::Mul,
                                    input(cb_gamma_tiles, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                                    DestReuseType::DEST_TO_SRCA>{},
                                pack);
                        } else {
                            eltwise_chain(
                                IterationShape::grid(nb, wc), BinaryFpu<BinaryFpuOp::Mul, x_in, rstd_in>{xr}, pack);
                        }
                    };
                    if constexpr (out_native) {
                        norm(PackTile<output(
                                 cb_output_tiles,
                                 ReservePolicy::None,
                                 PushPolicy::None,
                                 DataFormatReconfig::Enabled,
                                 TileAddressing::Strided)>{xr});
                    } else {
                        norm(PackTile<output(cb_output_tiles)>{});
                    }
                    if constexpr (has_gamma && !gamma_prescale) {
                        if (gamma_chunked) {
                            cb_pop_front(cb_gamma_tiles, gamma_chunk);
                        }
                    }
                }
                cb_pop_front(cb_rstd, nb);
            } else if constexpr (has_gamma && !gamma_prescale) {
                // fused: (x * rstd) in DEST, then * gamma (tiles hold gamma in every row: DEST reuse has no broadcast)
                eltwise_chain(
                    IterationShape::grid(nb, cw),
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(cb_x_resident),
                        input(
                            input(cb_rstd, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                            BroadcastDim::Col)>{},
                    DestReuseBinary<
                        BinaryFpuOp::Mul,
                        input(cb_gamma_tiles, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                        DestReuseType::DEST_TO_SRCA>{},
                    PackTile<output(cb_output_tiles)>{});
            } else {
                eltwise_chain(
                    IterationShape::grid(nb, cw),
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(cb_norm_src),
                        input(
                            input(cb_rstd, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                            BroadcastDim::Col)>{},
                    PackTile<output(cb_output_tiles)>{});
            }
        }
    };

    // R2 schedule: round b+lookahead's sumsq / prescale run before round b's combine waits on the exchange (the host
    // sizes cb_x_resident / cb_xg for 1 + lookahead blocks). One round: front(0), back(0) == the R1 schedule.
    // held: x tiles of the rounds still waiting for their normalize sit ahead of the new round's block in
    // cb_x_resident, unless the prescale already consumed them (sharded x is addressed absolutely).
    auto held_before = [&](uint32_t first, uint32_t last) {
        uint32_t held = 0;
        if constexpr (!x_sharded && !gamma_prescale) {
            for (uint32_t r = first; r < last; ++r) {
                held += rows_of(r) * cw;
            }
        }
        return held;
    };
    for (uint32_t round = 0; round < round_lookahead && round < num_blocks; ++round) {
        front(round, held_before(0, round));
    }
    for (uint32_t round = 0; round < num_blocks; ++round) {
        if (round + round_lookahead < num_blocks) {
            front(round + round_lookahead, held_before(round, round + round_lookahead));
        }
        back(round);
    }
    if constexpr (out_native) {
        cb_push_back(cb_output_tiles, core_rows * cw_phys);
    }
}
