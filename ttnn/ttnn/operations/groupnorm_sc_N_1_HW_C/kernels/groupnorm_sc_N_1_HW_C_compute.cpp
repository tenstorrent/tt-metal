// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — compute.
//
// pass 1 (per column group, inside ONE K-blocked matmul_block call whose PreKBlockFn runs the row chunks):
//   [RM] tilize -> square -> REDUCE_COL SUM of x and x^2 (Accumulate across row chunks into cb_colsum)
//   then  [S; Q](2 x cols) x E^T(cols x Kg)  K-block cg  ->  cb_partial (lane form, 2*Kg tiles)
// combine: root reduces the gathered records (REDUCE_COL over gather tiles) -> cb_totals_src;
//          every core: mean/rstd lane form -> broadcast to full tiles (cb_stats_g_full)
// pass 2 (per column group): per channel tile T: [mean; rstd]_full x E_T -> a_T = rstd_T*gamma_T,
//          b_T = beta_T - mean_T*a_T; then
//          y = x*a_T + b_T over every row chunk -> cb_out.
//
// Helper policy notes (caller-owned lifecycles the headers hand out):
//  * REDUCE_COL uses ReduceInputPolicy::WaitUpfrontNoPop + caller pop: the BulkWaitBulkPop path asserts the
//    input CB capacity to be a multiple of Ht*DEST_AUTO_LIMIT, which a cols_per_group < DEST_AUTO_LIMIT block
//    cannot satisfy; WaitUpfrontNoPop indexes the same row-major block and leaves the pop to us.
//  * cb_totals_recv / cb_stats_row / cb_x_pass2 (resident) are read with caller-managed (None, None) chain
//    policies + TileAddressing::Offset because their windows are random-access, multi-tile, or whole-block.
//
// Ragged blocks (op_design.md -> Work Distribution: "ragged last group/chunk keeps nominal push/pop counts and
// narrows only the work"): this core's Ht_core x Ct_core extents are RT args. Every CB quantum stays nominal
// (`chunk`, `cols`, `cols*Kg`) so no ring ever straddles its wrap point; the valid tiles of a block sit dense
// (row-major valid_rows x valid_cols) at the front of the quantum and only they are reduced / applied:
//  * x / xsq: dense valid tiles, `chunk` push/pop; the pad pages carry no data.
//  * colsum ring [S(cols) ; Q(cols)]: the reduce writes valid_cols tiles per statistic; the cols - valid_cols pad
//    slots are ZERO-filled (FillScalar chain) so the K = cols membership matmul (whose E^T pad rows are zero)
//    never multiplies 0 by uninitialised L1.
//  * membership / affine rows / a_full / b_full: valid_cols consumed or produced, the rest pad-popped / -pushed.
//  * RM input, ragged column group: tilize one tile-row at a time with the valid width, pad-pop the rest.

#include <stdint.h>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"
#include "groupnorm_sc_N_1_HW_C_ragged.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t ROLE_IDLE = 0;
constexpr uint32_t ROLE_ROOT = 2;

// tilize<W> with a runtime width w <= W_MAX (the tilize helper's block width is a template parameter).
template <uint32_t W_MAX, uint32_t cb_in, uint32_t cb_out>
FORCE_INLINE void tilize_width(uint32_t w, uint32_t num_blocks) {
    if constexpr (W_MAX > 1) {
        if (w != W_MAX) {
            tilize_width<W_MAX - 1, cb_in, cb_out>(w, num_blocks);
            return;
        }
    }
    ckl::tilize<W_MAX, cb_in, cb_out>(num_blocks);
}

// RM sticks of one (possibly ragged) block -> dense valid_rows x valid_cols tiles. A full-width group is the
// helper's multi-block call; a ragged column group is tilized one tile-row at a time (the reader pushes each
// tile-row as valid_cols data pages + a cols - valid_cols pad, keeping the cb_in ring aligned to tile-rows).
template <uint32_t cols, uint32_t cb_in, uint32_t cb_out>
FORCE_INLINE void tilize_ragged_block(uint32_t valid_rows, uint32_t valid_cols) {
    if (valid_cols == cols) {
        ckl::tilize<cols, cb_in, cb_out>(valid_rows);
        return;
    }
    for (uint32_t r = 0; r < valid_rows; ++r) {
        tilize_width<cols, cb_in, cb_out>(valid_cols, 1);
        groupnorm_ragged::pad_pop(cb_in, cols - valid_cols, cols - valid_cols);
    }
}
}  // namespace

void kernel_main() {
    // ---------------- compile-time args ----------------
    constexpr uint32_t cb_x_pass1 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x_pass2 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x_rm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_xsq = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t cb_colsum = get_compile_time_arg_val(5);
    constexpr uint32_t cb_membership = get_compile_time_arg_val(6);
    constexpr uint32_t cb_agg_interm = get_compile_time_arg_val(7);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(8);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(9);
    constexpr uint32_t cb_totals_src = get_compile_time_arg_val(10);
    constexpr uint32_t cb_totals_recv = get_compile_time_arg_val(11);
    constexpr uint32_t cb_stats_g_full = get_compile_time_arg_val(12);
    constexpr uint32_t cb_gamma_row = get_compile_time_arg_val(13);
    constexpr uint32_t cb_beta_row = get_compile_time_arg_val(14);
    constexpr uint32_t cb_stats_T = get_compile_time_arg_val(15);
    constexpr uint32_t cb_beta_full = get_compile_time_arg_val(16);  // transient beta_T broadcast to all rows
    constexpr uint32_t cb_a_full = get_compile_time_arg_val(17);
    constexpr uint32_t cb_b_full = get_compile_time_arg_val(18);
    constexpr uint32_t cb_out = get_compile_time_arg_val(19);
    constexpr uint32_t cb_stats_row = get_compile_time_arg_val(20);
    constexpr bool is_rm = get_compile_time_arg_val(21) != 0;
    constexpr bool resident = get_compile_time_arg_val(22) != 0;
    constexpr bool has_gamma = get_compile_time_arg_val(23) != 0;
    constexpr bool has_beta = get_compile_time_arg_val(24) != 0;
    constexpr uint32_t chunk_rows = get_compile_time_arg_val(25);
    constexpr uint32_t cols = get_compile_time_arg_val(26);
    constexpr uint32_t Kg = get_compile_time_arg_val(27);
    constexpr uint32_t gather_rows = get_compile_time_arg_val(28);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(29);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(30);
    constexpr uint32_t out_block = get_compile_time_arg_val(31);  // writer store block (divides chunk)
    constexpr uint32_t hw_tail = get_compile_time_arg_val(32);  // valid rows of the image's last tile-row (0 = aligned)

    // ---------------- runtime args ----------------
    const uint32_t image_count = get_arg_val<uint32_t>(0);
    const uint32_t role = get_arg_val<uint32_t>(1);
    const uint32_t inv_n_bits = get_arg_val<uint32_t>(2);
    const uint32_t eps_bits = get_arg_val<uint32_t>(3);
    const uint32_t Ht_core = get_arg_val<uint32_t>(4);
    const uint32_t Ct_core = get_arg_val<uint32_t>(5);
    const uint32_t owns_last_row = get_arg_val<uint32_t>(6);  // this core's rows end at the image's last tile-row

    constexpr uint32_t chunk = chunk_rows * cols;
    constexpr uint32_t num_stats = 2 * Kg;

    compute_kernel_hw_startup(cb_x_pass1, cb_scaler, cb_out);

    if (role == ROLE_IDLE) {
        return;
    }

    // Ragged accounting: blocks along each axis of this core's extent and the valid units of the last one.
    const auto row_axis = groupnorm_ragged::split(Ht_core, chunk_rows);
    const auto col_axis = groupnorm_ragged::split(Ct_core, cols);
    const uint32_t num_row_chunks = row_axis.count;
    const uint32_t num_col_groups = col_axis.count;
    const uint32_t blk = num_col_groups * num_row_chunks * chunk;  // nominal pages of the resident block

    using ckl::BinaryFpuOp;
    using ckl::BroadcastDim;
    using ckl::DataFormatReconfig;
    using ckl::DestReuseType;
    using ckl::Dst;
    using ckl::input;
    using ckl::InputTileMapping;
    using ckl::IterationShape;
    using ckl::output;
    using ckl::PopPolicy;
    using ckl::PushPolicy;
    using ckl::ReservePolicy;
    using ckl::TileAddressing;
    using ckl::WaitPolicy;

    CircularBuffer colsum_buf(cb_colsum);
    CircularBuffer membership_buf(cb_membership);
    CircularBuffer partial_buf(cb_partial);
    CircularBuffer agg_interm_buf(cb_agg_interm);
    CircularBuffer stats_g_full_buf(cb_stats_g_full);
    CircularBuffer stats_T_buf(cb_stats_T);

    // Ragged column group: keep the colsum ring at its nominal [S(cols) ; Q(cols)] layout. After a statistic's
    // valid_cols tiles were pushed, drop the previous chunk's pad tiles from the front (rc > 0) and push
    // cols - valid_cols ZERO tiles behind the new ones, so the K = cols membership matmul reads finite pads.
    auto pad_colsum_statistic = [&](uint32_t rc, uint32_t valid_cols) {
        const uint32_t pad = cols - valid_cols;
        if (pad == 0) {
            return;
        }
        if (rc > 0) {
            groupnorm_ragged::pad_pop(cb_colsum, pad, pad);
        }
        cb_reserve_back(cb_colsum, pad);
        ckl::eltwise_chain(
            IterationShape::tiles(pad),
            ckl::FillScalar<Dst::D0>{0.0f},
            ckl::PackTile<output(cb_colsum, ReservePolicy::None, PushPolicy::None), Dst::D0>{});
        cb_push_back(cb_colsum, pad);
    };

    // ---- pass-1 row-chunk work: ops 2..5 of the block schedule for one (valid_rows x valid_cols) block ----
    auto colsum_chunk = [&](uint32_t valid_rows, uint32_t valid_cols, uint32_t rc) {
        const uint32_t valid = valid_rows * valid_cols;
        // hw_non_aligned: the chunk holding the image's last tile-row reduces its last row tile (dense block ->
        // row valid_rows - 1 of every column) with the reader's partial scaler (tile 1 of cb_scaler), so the
        // padded rows >= HW contribute nothing whatever the input tensor carries there.
        const bool partial_last_row = (hw_tail != 0) && (owns_last_row != 0) && (rc + 1 == num_row_chunks);
        const auto partial_scaler =
            partial_last_row ? ckl::ReducePartialScaler::with_partial() : ckl::ReducePartialScaler::none();
        if constexpr (is_rm) {
            tilize_ragged_block<cols, cb_x_rm, cb_x_pass1>(valid_rows, valid_cols);
            groupnorm_ragged::pad_push(cb_x_pass1, chunk - valid, chunk);  // nominal chunk quantum
            if constexpr (resident) {
                cb_reserve_back(cb_x_pass2, chunk);  // pass-2 credit over the same (aliased) bytes
                cb_push_back(cb_x_pass2, chunk);
            }
        }
        // x^2 for the valid tiles (x stays fronted for the column sum); xsq keeps the nominal chunk quantum
        cb_reserve_back(cb_xsq, chunk);
        ckl::square<
            input(cb_x_pass1, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block),
            output(cb_xsq, ReservePolicy::None, PushPolicy::None)>(IterationShape::tiles(valid));
        cb_push_back(cb_xsq, chunk);
        // column sums of x and x^2 over valid_rows, accumulated across row chunks in cb_colsum = [S..; Q..]
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_COL,
            cb_x_pass1,
            cb_scaler,
            cb_colsum,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
            ckl::ReduceInputBlockShape::of(valid_rows, valid_cols),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::Accumulate::at(cb_colsum, rc),
            ckl::NoOp{},
            partial_scaler);
        pad_colsum_statistic(rc, valid_cols);
        cb_pop_front(cb_x_pass1, chunk);
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_COL,
            cb_xsq,
            cb_scaler,
            cb_colsum,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
            ckl::ReduceInputBlockShape::of(valid_rows, valid_cols),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::Accumulate::at(cb_colsum, rc),
            ckl::NoOp{},
            partial_scaler);
        pad_colsum_statistic(rc, valid_cols);
        cb_pop_front(cb_xsq, chunk);
    };

    // PreKBlockFn of the pass-1 matmul: the row chunks of column group cg (K-block cg).
    auto colsum_column_group = [&](uint32_t cg, uint32_t /*num_k_blocks*/, bool /*is_last*/) {
        const uint32_t valid_cols = col_axis.valid(cg, cols);
        for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
            colsum_chunk(row_axis.valid(rc, chunk_rows), valid_cols, rc);
        }
    };

    for (uint32_t img = 0; img < image_count; ++img) {
        // ================= pass 1: statistics =================
        // [S; Q] (M=2, K=cols) x E^T (K=cols, N=Kg), one K-block per column group, spill/reload via interm.
        ckl::matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/false,
            ckl::LastBlockTarget::Out,
            ckl::OutputCBLayout::SubblockMajor,
            ckl::matmul_config::InitMode::ShortAfterPreKBlock,
            ckl::InputPolicy::WaitAndPopPerKBlock,
            ckl::InputPolicy::WaitAndPopPerKBlock,
            ckl::NoPostCompute,
            decltype(colsum_column_group)>(
            colsum_buf,
            membership_buf,
            partial_buf,
            agg_interm_buf,
            ckl::MatmulBlockShape::of(2, in1_num_subblocks, 1, out_subblock_w, cols, num_col_groups),
            ckl::NoPostCompute{},
            colsum_column_group);

        // ================= combine =================
        if (role == ROLE_ROOT) {
            // rows = gather tiles per statistic (one record row per core), cols = the 2*Kg statistic tiles
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_COL,
                cb_gather,
                cb_scaler,
                cb_totals_src,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::of(gather_rows, num_stats));
            cb_pop_front(cb_gather, gather_rows * num_stats);
        }

        // finalize_stats: lane form [sum_g; sumsq_g] -> [mean_g; rstd_g] (row 0) -> full tiles
        cb_wait_front(cb_totals_recv, num_stats);
        cb_reserve_back(cb_stats_row, num_stats);
        ckl::eltwise_chain(
            IterationShape::tiles(Kg),
            ckl::CopyTile<input(cb_totals_recv, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block), Dst::D0>{},
            ckl::MulUnary<Dst::D0>{inv_n_bits},  // mean = sum / n
            ckl::CopyTile<
                input(
                    cb_totals_recv,
                    WaitPolicy::None,
                    PopPolicy::None,
                    InputTileMapping::Block,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Offset),
                Dst::D1>{Kg},
            ckl::MulUnary<Dst::D1>{inv_n_bits},                           // E[x^2]
            ckl::MulBinary<Dst::D0, Dst::D0, Dst::D2>{},                  // mean^2
            ckl::SubBinary<Dst::D1, Dst::D2, Dst::D1>{},                  // var = E[x^2] - mean^2
            ckl::Relu<Dst::D1>{},                                         // clamp cancellation below 0
            ckl::AddUnary<Dst::D1>{eps_bits},                             // var + eps
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::Off, Dst::D1>{},  // rstd
            ckl::PackTile<output(cb_stats_row, ReservePolicy::None, PushPolicy::None), Dst::D0>{},
            ckl::PackTile<
                output(
                    cb_stats_row,
                    ReservePolicy::None,
                    PushPolicy::None,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Offset),
                Dst::D1>{Kg});
        cb_push_back(cb_stats_row, num_stats);
        cb_pop_front(cb_totals_recv, num_stats);
        ckl::unary_bcast<
            BroadcastDim::Row,
            input(cb_stats_row, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
            output(cb_stats_g_full)>(IterationShape::tiles(num_stats));

        // ================= pass 2: apply =================
        if constexpr (resident) {
            cb_wait_front(cb_x_pass2, blk);
        }
        for (uint32_t cg = 0; cg < num_col_groups; ++cg) {
            const uint32_t valid_cols = col_axis.valid(cg, cols);
            // ---- build_affine_block: per channel tile T of this column group ----
            for (uint32_t tl = 0; tl < valid_cols; ++tl) {
                // [mean_full; rstd_full] (2 x Kg) x E_T (Kg x 1) -> cb_stats_T = [mean_T_full; rstd_T_full]
                ckl::matmul_block<
                    false,
                    false,
                    ckl::LastBlockTarget::Out,
                    ckl::OutputCBLayout::SubblockMajor,
                    ckl::matmul_config::InitMode::Short,
                    ckl::InputPolicy::WaitAndRetainOnLastBlock,
                    ckl::InputPolicy::WaitAndPopPerKBlock>(
                    stats_g_full_buf,
                    membership_buf,
                    stats_T_buf,
                    stats_T_buf,
                    ckl::MatmulBlockShape::of(2, 1, 1, 1, Kg, 1));
                cb_wait_front(cb_stats_T, 2);

                // a_T = rstd_T * gamma_T  (gamma row 0 broadcast down the tile), or rstd_T without gamma
                if constexpr (has_gamma) {
                    ckl::eltwise_chain(
                        IterationShape::one_tile(),
                        ckl::BinaryFpu<
                            BinaryFpuOp::Mul,
                            input(
                                cb_stats_T,
                                WaitPolicy::None,
                                PopPolicy::None,
                                InputTileMapping::Block,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Offset),
                            input(cb_gamma_row, BroadcastDim::Row, WaitPolicy::PerTile, PopPolicy::PerTile),
                            Dst::D0>{1u},
                        ckl::PackTile<output(cb_a_full), Dst::D0>{});
                } else {
                    ckl::eltwise_chain(
                        IterationShape::one_tile(),
                        ckl::CopyTile<
                            input(
                                cb_stats_T,
                                WaitPolicy::None,
                                PopPolicy::None,
                                InputTileMapping::Block,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Offset),
                            Dst::D0>{1u},
                        ckl::PackTile<output(cb_a_full), Dst::D0>{});
                }

                // b_T = beta_T - mean_T * a_T  (or -mean_T * a_T without beta): beta row 0 broadcast to a full
                // tile, then DEST = mean_T_full * a_T and b_T = beta_full - DEST (dest-reuse Sub, DEST_TO_SRCB).
                // Precision note: b and x*a are both ~|mean|*rstd*gamma and cancel in y = x*a + b; the FPU
                // evaluates them at tf32-class precision, so a |mean| >> std input loses ~ulp_tf32(|mean|*a)
                // absolute accuracy (the design's deferred shifted_two_pass_variance row would remove this).
                //
                // Ordering: the chain below reads a_T (tile tl of cb_a_full) with WaitPolicy::None, but that tile
                // was packed by the chain just above. Unpack, math and pack are separate threads; the ONLY thing
                // that orders "pack wrote a_T to L1" against "unpack reads a_T" is CB credit, so wait for it here.
                // Without this the unpacker can read stale L1 (non-deterministic per-element errors, seen on
                // RM input + gamma_only where no beta bcast sat between the two chains).
                cb_wait_front(cb_a_full, tl + 1);
                if constexpr (has_beta) {
                    ckl::unary_bcast<BroadcastDim::Row, input(cb_beta_row), output(cb_beta_full)>(
                        IterationShape::one_tile());
                    ckl::eltwise_chain(
                        IterationShape::one_tile(),
                        ckl::BinaryFpu<
                            BinaryFpuOp::Mul,
                            input(
                                cb_stats_T,
                                WaitPolicy::None,
                                PopPolicy::None,
                                InputTileMapping::Block,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Offset),
                            input(
                                cb_a_full,
                                BroadcastDim::None,
                                WaitPolicy::None,
                                PopPolicy::None,
                                InputTileMapping::Block,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Offset),
                            Dst::D0>{0u, tl},
                        ckl::DestReuseBinary<
                            BinaryFpuOp::Sub,
                            input(cb_beta_full, WaitPolicy::PerTile, PopPolicy::PerTile),
                            DestReuseType::DEST_TO_SRCB,
                            Dst::D0>{},
                        ckl::PackTile<output(cb_b_full), Dst::D0>{});
                } else {
                    ckl::eltwise_chain(
                        IterationShape::one_tile(),
                        ckl::BinaryFpu<
                            BinaryFpuOp::Mul,
                            input(
                                cb_stats_T,
                                WaitPolicy::None,
                                PopPolicy::None,
                                InputTileMapping::Block,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Offset),
                            input(
                                cb_a_full,
                                BroadcastDim::None,
                                WaitPolicy::None,
                                PopPolicy::None,
                                InputTileMapping::Block,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Offset),
                            Dst::D0>{0u, tl},
                        ckl::Negative<Dst::D0>{},
                        ckl::PackTile<output(cb_b_full), Dst::D0>{});
                }
                cb_pop_front(cb_stats_T, 2);
            }
            // Ragged last group: the reader pushed the nominal cols tiles of E / gamma / beta; drain the unused
            // pad tiles and pad a_full / b_full up to their nominal cols quantum (no data behind the pads).
            {
                const uint32_t pad = cols - valid_cols;
                groupnorm_ragged::pad_pop(cb_membership, pad * Kg, Kg);
                if constexpr (has_gamma) {
                    groupnorm_ragged::pad_pop(cb_gamma_row, pad, pad);
                }
                if constexpr (has_beta) {
                    groupnorm_ragged::pad_pop(cb_beta_row, pad, pad);
                }
                groupnorm_ragged::pad_push(cb_a_full, pad, pad);
                groupnorm_ragged::pad_push(cb_b_full, pad, pad);
            }

            // ---- apply_block: y = x * a_T + b_T over every row chunk of this column group ----
            auto apply_chunk = [&](uint32_t valid_rows, uint32_t rc) {
                const uint32_t valid = valid_rows * valid_cols;
                if constexpr (!resident) {
                    if constexpr (is_rm) {
                        tilize_ragged_block<cols, cb_x_rm, cb_x_pass2>(valid_rows, valid_cols);
                        groupnorm_ragged::pad_push(cb_x_pass2, chunk - valid, chunk);
                    }
                    cb_wait_front(cb_x_pass2, chunk);
                }
                const uint32_t x_base = resident ? (cg * num_row_chunks + rc) * chunk : 0u;
                ckl::eltwise_chain(
                    IterationShape::grid(valid_rows, valid_cols),
                    ckl::BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(
                            cb_x_pass2,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        input(
                            cb_a_full, BroadcastDim::None, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                        Dst::D0>{x_base},
                    // DEST_TO_SRCB (b -> srcA, DEST -> srcB): the LLK dest-reuse init unpacks the CB operand through
                    // unpacker A regardless of the reuse side, and the chain only re-programs srcA's format for
                    // DEST_TO_SRCB — with DEST_TO_SRCA a Float32 b_T after a bf16 x on srcA trips the format check.
                    // Add is commutative, so b + (x*a) is the same result.
                    ckl::DestReuseBinary<
                        BinaryFpuOp::Add,
                        input(cb_b_full, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                        DestReuseType::DEST_TO_SRCB,
                        Dst::D0>{},
                    ckl::PackTile<output(cb_out), Dst::D0>{});
                // the writer drains the nominal chunk per block in out_block groups; pad the unused pages
                groupnorm_ragged::pad_push(cb_out, chunk - valid, out_block);
                if constexpr (!resident) {
                    cb_pop_front(cb_x_pass2, chunk);
                }
            };
            for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
                apply_chunk(row_axis.valid(rc, chunk_rows), rc);
            }
            cb_pop_front(cb_a_full, cols);
            cb_pop_front(cb_b_full, cols);
        }
        if constexpr (resident) {
            cb_pop_front(cb_x_pass2, blk);
        }
        cb_pop_front(cb_stats_g_full, num_stats);
    }
}
