# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off (Perf round 2) for groupnorm_sc_N_1_HW_C: lane-form (row-0-only) pass-2 affine build + a
broadcast-row apply, deleting the full-tile broadcasts of the stats, gamma-side and beta-side operands.

Single core, everything resident in sharded L1, pure compute. The measured region is the whole post-finalize
chain of one image / one column group: c_stats_bcast + c_affine + c_apply (num_row_chunks chunks of
chunk_rows x cols tiles, the last chunk possibly ragged in rows).

Inputs (identical for every variant):
    cb_stats_row   [mean_row(Kg) ; rstd_row(Kg)]  Float32 lane-form tiles (row 0 = per-group value, rows 1..31 = 0
                   nominally; a `dirty_rows` host mode fills them with random finite junk to prove nothing reads them)
    cb_membership  E tiles Float32 0/1, k-major (kg * cols + tl) — the op's pass-2 reader order
    cb_gamma_row / cb_beta_row   bf16 row-0 tiles (rows 1..31 = 0)
    cb_x           bf16 x tiles, num_row_chunks quanta of `chunk` pages, valid tiles dense at the front of each quantum
    cb_zero        one bf16 zero tile (only method 4 reads it)
Output: cb_out — bf16 y tiles in the same layout as x (pad pages of a ragged chunk are pushed, never written).

method 0  batched      THE OP TODAY (baseline): unary_bcast<Row>(stats_row) -> cb_stats_g_full ; matmul
                       [mean;rstd]_full x E -> cb_stats_T ; a chain (rstd_T * gamma Row-bcast) -> cb_a (full) ;
                       unary_bcast<Row>(beta_row) -> cb_beta_full ; b chain (mean_T * a, DestReuse Sub beta_full)
                       -> cb_b (full) ; apply: per chunk eltwise_chain grid(valid_rows, cols) { BinaryFpu Mul
                       x * a_full (Row mapping), DestReuseBinary Add b_full (DEST_TO_SRCB), PackTile } (block 1).
method 1  batched_blk  as 0, apply chain with .block_size(cols) (one MOP init pair per tile-row, not per tile).
method 2  lane_fullab  lane build (no stats bcast: matmul straight on the lane-form stats; a_row = rstd_T_row *
                       gamma_row plain eltwise; b_row = beta_row - mean_T_row * a_row) then unary_bcast<Row> of
                       a_row / b_row -> full tiles and the method-0 apply. Helper-only graduation fallback.
method 3  lane_d2a     lane build ; apply per tile-row: mul_tiles_bcast_rows(x, a_row) into DEST, then RAW LLK
                       dest-reuse ELWADD<BroadcastType::ROW, DEST_TO_SRCA>(b_row): DEST -> srcA (srcA format set to
                       Float32 so the move keeps tf32-class precision), bcast_row(b_row) -> srcB, y = srcA + srcB.
method 4  lane_accadd  lane build ; apply per tile-row: mul_tiles_bcast_rows(x, a_row) into DEST, then RAW LLK
                       standard ELWADD<ROW> with acc_to_dest = 1: DEST += srcA(zero tile) + bcast_row(b_row). The
                       product never leaves the fp32 DEST (no re-route rounding) -> more precise than baseline.
method 5  lane_l1      lane build ; helper-only apply through an fp32 L1 round trip: chain x * bcast_row(a_row) ->
                       cb_interm ; chain interm + bcast_row(b_row) -> cb_out.

Precision contract (FIXED, never tuned): fp32_dest_acc_en=True, HiFi4, math_approx_mode=False, dst_full_sync_en=True;
stats / E / stats_T / a / b pages Float32, x / y / gamma / beta bf16 (the op's _statistic_page_dtype).

Why the LLK is needed for methods 3 / 4 (kernel_lib capability gaps, see README):
  * DestReuseBinary has no BroadcastDim (its LLK call is BroadcastType::NONE only), so "DEST + bcast_row(b)" is
    inexpressible through the chain; the LLK itself supports ROW + DEST_TO_SRCA (unpack_A MOP `unpack_srcb +
    dummy srcA dvalid`, math MOP `eltwise_binary_configure_mop_with_dest_reuse<ELWADD, ROW>`).
  * BinaryFpu's DestAccumulation passes acc_to_dest to the LLK init, but the bcast.h *_tiles_bcast wrappers and
    the LLK ELWMUL MOP hard-code acc_to_dest = 0; only ELWADD/ELWSUB honour it — so "DEST += x * bcast(a)" is not
    possible, while "DEST += 0 + bcast(b)" (method 4) is.
"""

import ttnn

TILE = 32
TILE_BYTES_F32 = 32 * 32 * 4
TILE_BYTES_BF16 = 32 * 32 * 2

# CB ids (compute-only, one core)
CB_STATS_ROW = 0  # input: lane-form [mean_row(Kg); rstd_row(Kg)] fp32
CB_MEMBERSHIP = 1  # input: E tiles fp32 0/1, k-major
CB_GAMMA_ROW = 2  # input: gamma row-0 tiles bf16
CB_BETA_ROW = 3  # input: beta row-0 tiles bf16
CB_X = 4  # input: x tiles bf16 (resident block)
CB_ZERO = 5  # input: one bf16 zero tile (method 4 srcA)
CB_STATS_G_FULL = 6  # scratch: stats broadcast to full tiles (methods 0/1)
CB_STATS_T = 7  # scratch: [mean_T(cols); rstd_T(cols)] fp32
CB_BETA_FULL = 8  # scratch: beta_T full tiles (methods 0/1)
CB_A = 9  # scratch: a (full tiles for 0/1, row tiles for 2..5) fp32, cols pages
CB_B = 10  # scratch: b fp32, cols pages
CB_A_FULL2 = 11  # scratch: a_row broadcast to full (method 2)
CB_B_FULL2 = 12  # scratch: b_row broadcast to full (method 2)
CB_INTERM = 13  # scratch: x * a fp32 round trip (method 5)
CB_OUT = 16  # output: y bf16

METHODS = {"batched": 0, "batched_blk": 1, "lane_fullab": 2, "lane_d2a": 3, "lane_accadd": 4, "lane_l1": 5}
VARIANTS = tuple(METHODS)
LANE_VARIANTS = ("lane_fullab", "lane_d2a", "lane_accadd", "lane_l1")
RAW_LLK_VARIANTS = ("lane_d2a", "lane_accadd")

# Precision contract of the op (FIXED — never tuned here; see groupnorm_sc_N_1_HW_C_program_descriptor.py)
MATH_FIDELITY = ttnn.MathFidelity.HiFi4
FP32_DEST_ACC_EN = True
DST_FULL_SYNC_EN = True
MATH_APPROX_MODE = False

_KERNEL = r"""
// groupnorm_sc_N_1_HW_C perf_experiments/affine_lane_form — post-finalize region (stats bcast + affine + apply),
// six implementations selected by the CT arg `method` (see affine_lane_form_bench.py).
//
// Helper bypass (methods 3 and 4 only, isolated bench): the apply's "+ b" step is written in raw LLK because the
// kernel_lib chain cannot express an FPU add of a ROW-BROADCAST operand against DEST:
//   * DestReuseBinary (chain.inl) takes a plain InputSpec and calls {add,sub,mul}_reuse_dest_tiles, which are
//     BroadcastType::NONE only — while the LLK dest-reuse MOP and the unpack_A MOP both support ROW + DEST_TO_SRCA
//     (method 3: llk_unpack_A<ROW, acc_to_dest=true, DEST_TO_SRCA> + llk_math_eltwise_binary<ELWADD, ROW, ...,
//     DEST_TO_SRCA>).
//   * bcast.h add_tiles_bcast_rows / add_bcast_rows_init hard-code acc_to_dest = 0; the LLK standard ELWADD MOP
//     honours it (method 4: llk_math_eltwise_binary_init<ELWADD, ROW, LoFi>(.., acc_to_dest = 1) +
//     llk_unpack_AB_init<ROW>, srcA fed with a zero tile so DEST += 0 + bcast_row(b)).
// Everything else (lane build, mul with a ROW-broadcast a_row, pack) is public compute API / kernel_lib.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_stats_row = get_compile_time_arg_val(0);
    constexpr uint32_t cb_membership = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_row = get_compile_time_arg_val(2);
    constexpr uint32_t cb_beta_row = get_compile_time_arg_val(3);
    constexpr uint32_t cb_x = get_compile_time_arg_val(4);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(5);
    constexpr uint32_t cb_stats_g_full = get_compile_time_arg_val(6);
    constexpr uint32_t cb_stats_T = get_compile_time_arg_val(7);
    constexpr uint32_t cb_beta_full = get_compile_time_arg_val(8);
    constexpr uint32_t cb_a = get_compile_time_arg_val(9);
    constexpr uint32_t cb_b = get_compile_time_arg_val(10);
    constexpr uint32_t cb_a_full2 = get_compile_time_arg_val(11);
    constexpr uint32_t cb_b_full2 = get_compile_time_arg_val(12);
    constexpr uint32_t cb_interm = get_compile_time_arg_val(13);
    constexpr uint32_t cb_out = get_compile_time_arg_val(14);
    constexpr uint32_t method = get_compile_time_arg_val(15);
    constexpr uint32_t cols = get_compile_time_arg_val(16);
    constexpr uint32_t Kg = get_compile_time_arg_val(17);
    constexpr uint32_t chunk_rows = get_compile_time_arg_val(18);
    constexpr uint32_t num_row_chunks = get_compile_time_arg_val(19);
    constexpr bool has_gamma = get_compile_time_arg_val(20) != 0;
    constexpr bool has_beta = get_compile_time_arg_val(21) != 0;
    constexpr uint32_t iters = get_compile_time_arg_val(22);
    constexpr uint32_t num_stats = 2 * Kg;
    constexpr uint32_t chunk = chunk_rows * cols;
    constexpr uint32_t x_pages = num_row_chunks * chunk;
    constexpr bool full_form = (method == 0 || method == 1);  // a / b are full tiles built from full-tile stats
    static_assert(cols <= ckl::DEST_AUTO_LIMIT, "a tile-row must fit DEST");

    const uint32_t Ht_core = get_arg_val<uint32_t>(0);  // valid tile-rows of the resident block (ragged last chunk)

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

    CircularBuffer stats_row_buf(cb_stats_row);
    CircularBuffer stats_g_full_buf(cb_stats_g_full);
    CircularBuffer membership_buf(cb_membership);
    CircularBuffer stats_T_buf(cb_stats_T);

    // sharded inputs are resident: make them visible once (retained across iterations)
    cb_reserve_back(cb_stats_row, num_stats);
    cb_push_back(cb_stats_row, num_stats);
    cb_reserve_back(cb_x, x_pages);
    cb_push_back(cb_x, x_pages);
    cb_reserve_back(cb_zero, 1);
    cb_push_back(cb_zero, 1);
    cb_wait_front(cb_stats_row, num_stats);
    cb_wait_front(cb_x, x_pages);
    cb_wait_front(cb_zero, 1);

    compute_kernel_hw_startup(cb_x, cb_a, cb_out);

    // ragged pad: keep the nominal `chunk` output pages per chunk (the op's writer drains nominal quanta)
    auto pad_push = [](uint32_t cb, uint32_t n) {
        if (n > 0) {
            cb_reserve_back(cb, n);
            cb_push_back(cb, n);
        }
    };

    for (uint32_t it = 0; it < iters; ++it) {
        // the consumed constants (E, gamma, beta) are re-credited for every iteration — same L1 bytes
        cb_reserve_back(cb_membership, cols * Kg);
        cb_push_back(cb_membership, cols * Kg);
        cb_reserve_back(cb_gamma_row, cols);
        cb_push_back(cb_gamma_row, cols);
        cb_reserve_back(cb_beta_row, cols);
        cb_push_back(cb_beta_row, cols);
        cb_wait_front(cb_membership, cols * Kg);
        cb_wait_front(cb_gamma_row, cols);
        cb_wait_front(cb_beta_row, cols);

        // ================= c_stats_bcast (full form only) =================
        if constexpr (full_form) {
            MaybeDeviceZoneScope("c_stats_bcast");
            ckl::unary_bcast<
                BroadcastDim::Row,
                input(cb_stats_row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block),
                output(cb_stats_g_full)>(IterationShape::tiles(num_stats));
        }

        // ================= c_affine =================
        {
            MaybeDeviceZoneScope("c_affine");
            // [mean; rstd] (2 x Kg) x E (Kg x cols) -> cb_stats_T = [mean_T0..; rstd_T0..] (SubblockMajor, 1 x cols).
            // Full form: in0 = the broadcast full tiles. Lane form: in0 = the lane-form stats themselves — row 0 of
            // the output is row 0 of in0 times E, exactly the per-channel stats; rows 1..31 are never read.
            ckl::matmul_block<
                false,
                false,
                ckl::LastBlockTarget::Out,
                ckl::OutputCBLayout::SubblockMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndRetainOnLastBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock>(
                full_form ? stats_g_full_buf : stats_row_buf,
                membership_buf,
                stats_T_buf,
                stats_T_buf,
                ckl::MatmulBlockShape::of(2, 1, 1, cols, Kg, 1));
            cb_wait_front(cb_stats_T, 2 * cols);

            // a = rstd_T * gamma (rstd tiles at offset cols). Full form: gamma row broadcast down the tile.
            // Lane form: plain eltwise of two row-0 tiles (rows 1..31: 0 * 0).
            if constexpr (has_gamma) {
                if constexpr (full_form) {
                    ckl::eltwise_chain(
                        IterationShape::tiles(cols),
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
                            Dst::D0>{cols},
                        ckl::PackTile<output(cb_a), Dst::D0>{});
                } else {
                    ckl::eltwise_chain(
                        IterationShape::tiles(cols),
                        ckl::BinaryFpu<
                            BinaryFpuOp::Mul,
                            input(
                                cb_stats_T,
                                WaitPolicy::None,
                                PopPolicy::None,
                                InputTileMapping::Block,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Offset),
                            input(cb_gamma_row, BroadcastDim::None, WaitPolicy::PerTile, PopPolicy::PerTile),
                            Dst::D0>{cols},
                        ckl::PackTile<output(cb_a), Dst::D0>{});
                }
            } else {
                ckl::eltwise_chain(
                    IterationShape::tiles(cols),
                    ckl::CopyTile<
                        input(
                            cb_stats_T,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        Dst::D0>{cols},
                    ckl::PackTile<output(cb_a), Dst::D0>{});
                // gamma rows are unused: drop the credit (constants are still pushed every iteration)
                cb_pop_front(cb_gamma_row, cols);
            }
            // pack -> unpack ordering of a is CB credit only: wait for the whole group before reading it back
            cb_wait_front(cb_a, cols);

            // b = beta - mean_T * a  (or -mean_T * a without beta)
            if constexpr (has_beta) {
                if constexpr (full_form) {
                    ckl::unary_bcast<
                        BroadcastDim::Row,
                        input(cb_beta_row, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                        output(cb_beta_full)>(IterationShape::tiles(cols));
                    ckl::eltwise_chain(
                        IterationShape::tiles(cols),
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
                                cb_a,
                                BroadcastDim::None,
                                WaitPolicy::None,
                                PopPolicy::None,
                                InputTileMapping::Block,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Offset),
                            Dst::D0>{0u, 0u},
                        ckl::DestReuseBinary<
                            BinaryFpuOp::Sub,
                            input(cb_beta_full, WaitPolicy::PerTile, PopPolicy::PerTile),
                            DestReuseType::DEST_TO_SRCB,
                            Dst::D0>{},
                        ckl::PackTile<output(cb_b), Dst::D0>{});
                } else {
                    // lane: beta_row (bf16 row-0 tile) straight into srcA of the dest-reuse Sub — no broadcast
                    ckl::eltwise_chain(
                        IterationShape::tiles(cols),
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
                                cb_a,
                                BroadcastDim::None,
                                WaitPolicy::None,
                                PopPolicy::None,
                                InputTileMapping::Block,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Offset),
                            Dst::D0>{0u, 0u},
                        ckl::DestReuseBinary<
                            BinaryFpuOp::Sub,
                            input(cb_beta_row, WaitPolicy::PerTile, PopPolicy::PerTile),
                            DestReuseType::DEST_TO_SRCB,
                            Dst::D0>{},
                        ckl::PackTile<output(cb_b), Dst::D0>{});
                }
            } else {
                ckl::eltwise_chain(
                    IterationShape::tiles(cols),
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
                            cb_a,
                            BroadcastDim::None,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        Dst::D0>{0u, 0u},
                    ckl::Negative<Dst::D0>{},
                    ckl::PackTile<output(cb_b), Dst::D0>{});
                // beta rows are unused: drop the credit like the op's has_beta=false path (constants still pushed)
                cb_pop_front(cb_beta_row, cols);
            }
            cb_pop_front(cb_stats_T, 2 * cols);

            if constexpr (method == 2) {
                // lane_fullab: materialise the full a / b tiles from the row tiles (2 * cols bcasts)
                cb_wait_front(cb_b, cols);
                ckl::unary_bcast<
                    BroadcastDim::Row,
                    input(cb_a, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                    output(cb_a_full2)>(IterationShape::tiles(cols));
                ckl::unary_bcast<
                    BroadcastDim::Row,
                    input(cb_b, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                    output(cb_b_full2)>(IterationShape::tiles(cols));
            }
        }  // c_affine

        // ================= c_apply =================
        constexpr uint32_t cb_af = (method == 2) ? cb_a_full2 : cb_a;
        constexpr uint32_t cb_bf = (method == 2) ? cb_b_full2 : cb_b;

        for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
            const uint32_t rows_left = Ht_core - rc * chunk_rows;
            const uint32_t valid_rows = rows_left < chunk_rows ? rows_left : chunk_rows;
            const uint32_t x_base = rc * chunk;
            MaybeDeviceZoneScope("c_apply");
            if constexpr (method == 0 || method == 2) {
                // the op's apply chain, verbatim (block_size 1: both MOP inits re-issued per tile)
                ckl::eltwise_chain(
                    IterationShape::grid(valid_rows, cols),
                    ckl::BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(
                            cb_x,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        input(cb_af, BroadcastDim::None, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                        Dst::D0>{x_base},
                    ckl::DestReuseBinary<
                        BinaryFpuOp::Add,
                        input(cb_bf, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                        DestReuseType::DEST_TO_SRCB,
                        Dst::D0>{},
                    ckl::PackTile<output(cb_out), Dst::D0>{});
            } else if constexpr (method == 1) {
                ckl::eltwise_chain(
                    IterationShape::grid(valid_rows, cols).block_size(cols),
                    ckl::BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(
                            cb_x,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        input(cb_af, BroadcastDim::None, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                        Dst::D0>{x_base},
                    ckl::DestReuseBinary<
                        BinaryFpuOp::Add,
                        input(cb_bf, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                        DestReuseType::DEST_TO_SRCB,
                        Dst::D0>{},
                    ckl::PackTile<output(cb_out), Dst::D0>{});
            } else if constexpr (method == 5) {
                // helper-only L1 round trip: x * bcast_row(a_row) -> cb_interm (fp32) ; interm + bcast_row(b_row) -> y
                ckl::eltwise_chain(
                    IterationShape::grid(valid_rows, cols),
                    ckl::BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(
                            cb_x,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        input(
                            cb_a,
                            BroadcastDim::Row,
                            WaitPolicy::Upfront,
                            PopPolicy::None,
                            InputTileMapping::Row,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Direct),
                        Dst::D0>{x_base},
                    ckl::PackTile<output(cb_interm), Dst::D0>{});
                ckl::eltwise_chain(
                    IterationShape::grid(valid_rows, cols),
                    ckl::BinaryFpu<
                        BinaryFpuOp::Add,
                        input(cb_interm, WaitPolicy::PerTile, PopPolicy::PerTile, DataFormatReconfig::Enabled),
                        input(
                            cb_b,
                            BroadcastDim::Row,
                            WaitPolicy::Upfront,
                            PopPolicy::None,
                            InputTileMapping::Row,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Direct),
                        Dst::D0>{},
                    ckl::PackTile<output(cb_out), Dst::D0>{});
            } else {
                // ===== methods 3 / 4: raw apply, one tile-row (cols tiles <= DEST) per DEST window =====
                cb_wait_front(cb_a, cols);
                cb_wait_front(cb_b, cols);
                pack_reconfig_data_format(cb_out);
                for (uint32_t r = 0; r < valid_rows; ++r) {
                    tile_regs_acquire();
                    // D[c] = x[r, c] * bcast_row(a_row[c])   (public API: srcA <- x bf16, srcB <- a_row fp32)
                    reconfig_data_format(cb_x, cb_a);
                    mul_bcast_rows_init(cb_x, cb_a);
                    for (uint32_t c = 0; c < cols; ++c) {
                        mul_tiles_bcast_rows(cb_x, cb_a, x_base + r * cols + c, c, c);
                    }
                    if constexpr (method == 3) {
                        // D[c] = D[c] + bcast_row(b_row[c]) — dest-reuse ELWADD with a ROW-broadcast srcB
                        // (DEST -> srcA). srcA's format is switched to the fp32 CB's so the MOVD2A conversion is
                        // tf32-class (a bf16 srcA — x's format — would round the product to 8 mantissa bits).
                        reconfig_data_format_srca(cb_x, cb_b);
                        reconfig_data_format_srcb(cb_a, cb_b);
                        UNPACK((llk_unpack_A_init<
                                ckernel::BroadcastType::ROW,
                                true /*acc_to_dest: dest-reuse handshake*/,
                                ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(false, false, cb_b)));
                        MATH((llk_math_eltwise_binary_init<
                              ckernel::EltwiseBinaryType::ELWADD,
                              ckernel::BroadcastType::ROW,
                              MathFidelity::LoFi,
                              ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_b, cb_b, 0 /*acc_to_dest*/)));
                        for (uint32_t c = 0; c < cols; ++c) {
                            UNPACK((llk_unpack_A<
                                    ckernel::BroadcastType::ROW,
                                    true,
                                    ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_b, c)));
                            MATH((llk_math_eltwise_binary<
                                  ckernel::EltwiseBinaryType::ELWADD,
                                  ckernel::BroadcastType::ROW,
                                  DST_ACCUM_MODE,
                                  MathFidelity::LoFi,
                                  ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                                cb_b, cb_b, c, false /*clear_fp32_dst_acc*/)));
                        }
                    } else {
                        // D[c] += srcA(zero tile) + bcast_row(b_row[c]) — standard ELWADD with acc_to_dest = 1;
                        // the fp32 product stays in DEST (no re-route rounding). srcA <- cb_zero (bf16, same
                        // format as x), srcB <- b_row (fp32).
                        reconfig_data_format_srca(cb_x, cb_zero);
                        reconfig_data_format_srcb(cb_a, cb_b);
                        MATH((llk_math_eltwise_binary_init<
                              ckernel::EltwiseBinaryType::ELWADD,
                              ckernel::BroadcastType::ROW,
                              MathFidelity::LoFi>(cb_zero, cb_b, 1 /*acc_to_dest*/)));
                        UNPACK((llk_unpack_AB_init<ckernel::BroadcastType::ROW>(cb_zero, cb_b)));
                        for (uint32_t c = 0; c < cols; ++c) {
                            MATH((llk_math_eltwise_binary<
                                  ckernel::EltwiseBinaryType::ELWADD,
                                  ckernel::BroadcastType::ROW,
                                  DST_ACCUM_MODE,
                                  MathFidelity::LoFi,
                                  ckernel::EltwiseBinaryReuseDestType::NONE>(cb_zero, cb_b, c, false)));
                            UNPACK((llk_unpack_AB<ckernel::BroadcastType::ROW>(cb_zero, cb_b, 0, c)));
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_out, cols);
                    for (uint32_t c = 0; c < cols; ++c) {
                        pack_tile(c, cb_out, c);
                    }
                    cb_push_back(cb_out, cols);
                    tile_regs_release();
                }
            }
            pad_push(cb_out, chunk - valid_rows * cols);
        }

        // ---- end of image / column group: drop the consumed affine tiles (as the op does per column group) ----
        cb_wait_front(cb_af, cols);
        cb_pop_front(cb_af, cols);
        cb_wait_front(cb_bf, cols);
        cb_pop_front(cb_bf, cols);
        if constexpr (full_form) {
            cb_pop_front(cb_stats_g_full, num_stats);
        }
        if (it + 1 < iters) {
            cb_wait_front(cb_out, x_pages);
            cb_pop_front(cb_out, x_pages);
        }
    }
    cb_pop_front(cb_stats_row, num_stats);
    cb_pop_front(cb_x, x_pages);
    cb_pop_front(cb_zero, 1);
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def sharded_row_config(num_tiles):
    """One tile-row of `num_tiles` tiles on core (0,0) — page t is column block t."""
    return ttnn.create_sharded_memory_config(
        shape=(TILE, num_tiles * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id, num_pages, dtype, page_bytes):
    return ttnn.CBDescriptor(
        total_size=page_bytes * max(1, num_pages),
        core_ranges=_single_core(),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_bytes)],
    )


def compute_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=MATH_FIDELITY,
        math_approx_mode=MATH_APPROX_MODE,
        fp32_dest_acc_en=FP32_DEST_ACC_EN,
        dst_full_sync_en=DST_FULL_SYNC_EN,
    )


def membership_tile_index(tl, kg, cols, Kg):
    """Page index of E tile (channel tile tl, group tile kg): the matmul helper's row-major K x N order."""
    return kg * cols + tl


def create_program_descriptor(
    tensors, *, variant, cols, Kg, chunk_rows, num_row_chunks, Ht_core, has_gamma, has_beta, iters=1, zones=False
):
    method = METHODS[variant]
    ct = [
        CB_STATS_ROW,
        CB_MEMBERSHIP,
        CB_GAMMA_ROW,
        CB_BETA_ROW,
        CB_X,
        CB_ZERO,
        CB_STATS_G_FULL,
        CB_STATS_T,
        CB_BETA_FULL,
        CB_A,
        CB_B,
        CB_A_FULL2,
        CB_B_FULL2,
        CB_INTERM,
        CB_OUT,
        method,
        cols,
        Kg,
        chunk_rows,
        num_row_chunks,
        int(has_gamma),
        int(has_beta),
        iters,
    ]
    defines = [] if zones else [("KERNEL_LIB_PERF_ZONES_OFF", "1")]
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [Ht_core]
    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=ct,
        runtime_args=rt,
        defines=defines,
        config=compute_config(),
    )
    full_form = method in (0, 1)
    chunk = chunk_rows * cols
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_STATS_ROW, tensors["stats_row"]),
        ttnn.cb_descriptor_from_sharded_tensor(CB_MEMBERSHIP, tensors["membership"]),
        ttnn.cb_descriptor_from_sharded_tensor(CB_GAMMA_ROW, tensors["gamma"]),
        ttnn.cb_descriptor_from_sharded_tensor(CB_BETA_ROW, tensors["beta"]),
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, tensors["x"]),
        ttnn.cb_descriptor_from_sharded_tensor(CB_ZERO, tensors["zero"]),
        _scratch_cb(CB_STATS_G_FULL, 2 * Kg if full_form else 0, ttnn.float32, TILE_BYTES_F32),
        _scratch_cb(CB_STATS_T, 2 * cols, ttnn.float32, TILE_BYTES_F32),
        _scratch_cb(CB_BETA_FULL, cols if full_form else 0, ttnn.float32, TILE_BYTES_F32),
        _scratch_cb(CB_A, cols, ttnn.float32, TILE_BYTES_F32),
        _scratch_cb(CB_B, cols, ttnn.float32, TILE_BYTES_F32),
        _scratch_cb(CB_A_FULL2, cols if method == 2 else 0, ttnn.float32, TILE_BYTES_F32),
        _scratch_cb(CB_B_FULL2, cols if method == 2 else 0, ttnn.float32, TILE_BYTES_F32),
        _scratch_cb(CB_INTERM, chunk if method == 5 else 0, ttnn.float32, TILE_BYTES_F32),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, tensors["y"]),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run_region(
    device, inputs, *, variant, cols, Kg, chunk_rows, num_row_chunks, Ht_core, has_gamma, has_beta, iters=1, zones=False
):
    """Allocate y (bf16, num_row_chunks*chunk tiles) and run one variant of the whole region. Returns y."""
    n_pages = num_row_chunks * chunk_rows * cols
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, n_pages * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, sharded_row_config(n_pages)
    )
    tensors = dict(inputs)
    tensors["y"] = y
    desc = create_program_descriptor(
        tensors,
        variant=variant,
        cols=cols,
        Kg=Kg,
        chunk_rows=chunk_rows,
        num_row_chunks=num_row_chunks,
        Ht_core=Ht_core,
        has_gamma=has_gamma,
        has_beta=has_beta,
        iters=iters,
        zones=zones,
    )
    ttnn.generic_op(
        [inputs["stats_row"], inputs["membership"], inputs["gamma"], inputs["beta"], inputs["x"], inputs["zero"], y],
        desc,
    )
    return y
