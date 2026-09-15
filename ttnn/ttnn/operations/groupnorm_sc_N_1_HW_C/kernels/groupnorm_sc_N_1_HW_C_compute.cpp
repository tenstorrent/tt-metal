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

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t ROLE_IDLE = 0;
constexpr uint32_t ROLE_ROOT = 2;
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
    constexpr uint32_t num_col_groups = get_compile_time_arg_val(28);
    constexpr uint32_t num_row_chunks = get_compile_time_arg_val(29);
    constexpr uint32_t gather_rows = get_compile_time_arg_val(30);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(31);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(32);

    // ---------------- runtime args ----------------
    const uint32_t image_count = get_arg_val<uint32_t>(0);
    const uint32_t role = get_arg_val<uint32_t>(1);
    const uint32_t inv_n_bits = get_arg_val<uint32_t>(2);
    const uint32_t eps_bits = get_arg_val<uint32_t>(3);

    constexpr uint32_t chunk = chunk_rows * cols;
    constexpr uint32_t num_stats = 2 * Kg;
    constexpr uint32_t blk = num_col_groups * num_row_chunks * chunk;

    compute_kernel_hw_startup(cb_x_pass1, cb_scaler, cb_out);

    if (role == ROLE_IDLE) {
        return;
    }

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

    // ---- pass-1 row-chunk work: ops 2..5 of the block schedule for column group cg ----
    auto colsum_column_group = [&](uint32_t /*cg*/, uint32_t /*num_k_blocks*/, bool /*is_last*/) {
        for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
            if constexpr (is_rm) {
                ckl::tilize<cols, cb_x_rm, cb_x_pass1>(chunk_rows);
                if constexpr (resident) {
                    cb_reserve_back(cb_x_pass2, chunk);  // pass-2 credit over the same (aliased) bytes
                    cb_push_back(cb_x_pass2, chunk);
                }
            }
            // x^2 for the chunk (x stays fronted for the column sum)
            ckl::square<
                input(cb_x_pass1, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block),
                output(cb_xsq)>(IterationShape::tiles(chunk));
            // column sums of x and x^2 over chunk_rows, accumulated across row chunks in cb_colsum = [S..; Q..]
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_COL,
                cb_x_pass1,
                cb_scaler,
                cb_colsum,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
                ckl::ReduceInputBlockShape::of(chunk_rows, cols),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(cb_colsum, rc));
            cb_pop_front(cb_x_pass1, chunk);
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_COL,
                cb_xsq,
                cb_scaler,
                cb_colsum,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
                ckl::ReduceInputBlockShape::of(chunk_rows, cols),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(cb_colsum, rc));
            cb_pop_front(cb_xsq, chunk);
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
            // ---- build_affine_block: per channel tile T of this column group ----
            for (uint32_t tl = 0; tl < cols; ++tl) {
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

            // ---- apply_block: y = x * a_T + b_T over every row chunk of this column group ----
            for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
                if constexpr (!resident) {
                    if constexpr (is_rm) {
                        ckl::tilize<cols, cb_x_rm, cb_x_pass2>(chunk_rows);
                    }
                    cb_wait_front(cb_x_pass2, chunk);
                }
                const uint32_t x_base = resident ? (cg * num_row_chunks + rc) * chunk : 0u;
                ckl::eltwise_chain(
                    IterationShape::grid(chunk_rows, cols),
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
                if constexpr (!resident) {
                    cb_pop_front(cb_x_pass2, chunk);
                }
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
