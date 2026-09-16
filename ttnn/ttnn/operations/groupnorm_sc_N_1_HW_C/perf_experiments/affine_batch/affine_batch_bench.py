# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for idea E1 of groupnorm_sc_N_1_HW_C: batch the pass-2 affine build per column group.

Single core, everything resident in sharded L1, pure compute (no NoC movement): the measured delta is the
per-helper-call overhead the per-tile loop pays (init + data-format reconfig + pipeline fill/drain per call).

What the op does today (baseline, method 0) for one column group of `cols` channel tiles, PER TILE T:
    matmul_block  [mean_full; rstd_full](2 x Kg) x E_T(Kg x 1)  -> cb_stats_T = [mean_T; rstd_T]     (fp32 full tiles)
    chain         a_T = rstd_T * gamma_T          (gamma row-0 tile, BroadcastDim::Row on srcB)     -> cb_a_full
    unary_bcast   beta_T row 0 -> full tile                                                         -> cb_beta_full
    chain         b_T = beta_full - mean_T * a_T  (BinaryFpu Mul + DestReuseBinary Sub DEST_TO_SRCB) -> cb_b_full
i.e. 4 helper calls per tile.

Candidates (all under the SAME precision contract: fp32 DEST, HiFi4, approx off, full sync; stats fp32; gamma/beta bf16):
    method 1  batched        one matmul_block with N = cols (out block 2 x cols: row 0 = mean_T for every T, row 1 =
                             rstd_T), one a-chain over cols tiles, one unary_bcast over cols beta rows, one b-chain
                             over cols tiles. Same math per element as the baseline -> bit-identical.
    method 2  batched+fold   as method 1 but no cb_beta_full round trip: the b-chain is
                             D0 = mean_T * a_T (FPU), D1 = bcast_row(beta_T) (UnaryBcast element), D0 = D1 - D0 (SFPU
                             SubBinary). The FPU sub with a Row-broadcast operand against DEST is inexpressible
                             (DestReuseBinary takes a plain InputSpec; the LLK binary_dest_reuse path is
                             BroadcastType::NONE only), so the subtraction moves to the SFPU (fp32-exact instead of
                             the FPU's tf32-class operands -> not bit-identical, slightly MORE precise).
    method 3  fused          one matmul_block + ONE chain producing a and b for all cols tiles with no intermediate CB
                             at all: D0 = rstd*gamma (a), D1 = rstd*gamma, D1 = mean * D1 (DestReuseBinary Mul,
                             CB->srcA, DEST->srcB), D2 = bcast_row(beta), D1 = D2 - D1 (SFPU), pack D0 -> a_full,
                             D1 -> b_full. Removes the a_full pack->unpack ordering hazard entirely (a never leaves
                             DEST). Costs one extra FPU mul per tile.

E tile order: the matmul helper indexes in1 as k * N + n (row-major K x N, matmul_block_helpers.inl in1_index +=
in1_per_core_w). The op's reader writes pass-2 membership tiles N-major (tl * Kg + kg); with N = 1 per call (baseline)
that is the same thing. The batched matmul needs kg * cols + tl, which coincides with the reader's order whenever
Kg == 1 (every G <= 32 shape). The host here lays E out per method.
"""

import math

import ttnn

TILE = 32
TILE_BYTES_F32 = 32 * 32 * 4
TILE_BYTES_BF16 = 32 * 32 * 2

# CB ids (compute-only, one core)
CB_STATS_G_FULL = 0  # [mean_full(Kg) ; rstd_full(Kg)] fp32, all rows equal (input)
CB_MEMBERSHIP = 1  # E tiles fp32 0/1 (input)
CB_GAMMA_ROW = 2  # gamma row-0 tiles bf16 (input)
CB_BETA_ROW = 3  # beta row-0 tiles bf16 (input)
CB_STATS_T = 4  # scratch: [mean_T ; rstd_T] fp32 (2 pages baseline, 2*cols batched)
CB_BETA_FULL = 5  # scratch: beta_T broadcast to a full tile (1 page baseline, cols batched; unused by methods 2/3)
CB_A_FULL = 16  # output a_T fp32 full tiles (cols)
CB_B_FULL = 17  # output b_T fp32 full tiles (cols)

METHODS = {"baseline": 0, "batched": 1, "batched_fold": 2, "fused": 3, "fused_sfpu": 4}
VARIANTS = tuple(METHODS)

# Precision contract of the op (FIXED — never tuned here; see groupnorm_sc_N_1_HW_C_program_descriptor.py)
MATH_FIDELITY = ttnn.MathFidelity.HiFi4
FP32_DEST_ACC_EN = True
DST_FULL_SYNC_EN = True
MATH_APPROX_MODE = False

_KERNEL = r"""
// groupnorm_sc_N_1_HW_C perf_experiments/affine_batch — pass-2 affine build, four implementations (CT arg `method`).
// Helper usage only (no raw LLK): matmul_block, eltwise_chain, unary_bcast + chain elements BinaryFpu /
// DestReuseBinary / UnaryBcast / SubBinary(SFPU) / PackTile.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_stats_g_full = get_compile_time_arg_val(0);
    constexpr uint32_t cb_membership = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_row = get_compile_time_arg_val(2);
    constexpr uint32_t cb_beta_row = get_compile_time_arg_val(3);
    constexpr uint32_t cb_stats_T = get_compile_time_arg_val(4);
    constexpr uint32_t cb_beta_full = get_compile_time_arg_val(5);
    constexpr uint32_t cb_a_full = get_compile_time_arg_val(6);
    constexpr uint32_t cb_b_full = get_compile_time_arg_val(7);
    constexpr uint32_t method = get_compile_time_arg_val(8);
    constexpr uint32_t cols = get_compile_time_arg_val(9);
    constexpr uint32_t Kg = get_compile_time_arg_val(10);
    constexpr uint32_t iters = get_compile_time_arg_val(11);
    constexpr uint32_t num_stats = 2 * Kg;

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

    CircularBuffer stats_g_full_buf(cb_stats_g_full);
    CircularBuffer membership_buf(cb_membership);
    CircularBuffer stats_T_buf(cb_stats_T);

    // sharded inputs are resident: make them visible once (stats retained across iterations)
    cb_reserve_back(cb_stats_g_full, num_stats);
    cb_push_back(cb_stats_g_full, num_stats);

    // same boot as the op (default SrcOrder; the matmul helper reconfigs + inits per call)
    compute_kernel_hw_startup(cb_stats_g_full, cb_membership, cb_a_full);

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

        if constexpr (method == 0) {
            // ================= baseline: the op's per-tile sequence, verbatim =================
            MaybeDeviceZoneScope("ab_baseline");
            for (uint32_t tl = 0; tl < cols; ++tl) {
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

                cb_wait_front(cb_a_full, tl + 1);
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
                cb_pop_front(cb_stats_T, 2);
            }
        } else {
            // ================= batched: one call per stage for the whole column group =================
            MaybeDeviceZoneScope("ab_batched");
            // [mean_full; rstd_full](2 x Kg) x E(Kg x cols) -> cb_stats_T: tiles 0..cols-1 = mean_T, cols..2cols-1 = rstd_T
            // (SubblockMajor with in0_num_subblocks = 2, out_subblock 1 x cols; in1 tile order kg * cols + tl)
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
                ckl::MatmulBlockShape::of(2, 1, 1, cols, Kg, 1));
            cb_wait_front(cb_stats_T, 2 * cols);

            if constexpr (method == 1 || method == 2) {
                // a_T = rstd_T * gamma_T for every T (rstd tiles at offset cols; gamma streamed one row tile per T)
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
                    ckl::PackTile<output(cb_a_full), Dst::D0>{});
                // pack -> unpack ordering of a_T is CB credit only: wait for the whole group before reading it back
                cb_wait_front(cb_a_full, cols);
            }

            if constexpr (method == 1) {
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
                            cb_a_full,
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
                    ckl::PackTile<output(cb_b_full), Dst::D0>{});
            } else if constexpr (method == 2) {
                // b_T = bcast_row(beta_T) - mean_T * a_T without the beta_full round trip: the bcast lands in D1 and
                // the subtraction runs on the SFPU (DEST - DEST); the FPU dest-reuse Sub cannot take a Row-broadcast
                // operand (helper + LLK gap, see module docstring).
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
                            cb_a_full,
                            BroadcastDim::None,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        Dst::D0>{0u, 0u},
                    ckl::UnaryBcast<
                        BroadcastDim::Row,
                        input(cb_beta_row, WaitPolicy::PerTile, PopPolicy::PerTile),
                        Dst::D1>{},
                    ckl::SubBinary<Dst::D1, Dst::D0, Dst::D0>{},
                    ckl::PackTile<output(cb_b_full), Dst::D0>{});
            } else if constexpr (method == 4) {
                // fused_sfpu: a stays in DEST; b = bcast_row(beta) - mean * a with the product and the difference on
                // the SFPU (fp32 DEST-DEST), only the a = rstd*gamma product on the FPU (as the baseline).
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
                    ckl::CopyTile<
                        input(
                            cb_stats_T,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        Dst::D1>{0u},
                    ckl::MulBinary<Dst::D1, Dst::D0, Dst::D1>{},
                    ckl::UnaryBcast<
                        BroadcastDim::Row,
                        input(cb_beta_row, WaitPolicy::PerTile, PopPolicy::PerTile),
                        Dst::D2>{},
                    ckl::SubBinary<Dst::D2, Dst::D1, Dst::D1>{},
                    ckl::PackTile<output(cb_a_full), Dst::D0>{},
                    ckl::PackTile<output(cb_b_full), Dst::D1>{});
            } else {
                // fused: a and b for every T in ONE chain, nothing leaves DEST until the two packs.
                //   D0 = rstd_T * gamma_T                       (a_T, packed to cb_a_full)
                //   D1 = rstd_T * gamma_T ; D1 = mean_T * D1    (DestReuseBinary Mul: CB mean_T -> srcA, DEST -> srcB)
                //   D2 = bcast_row(beta_T) ; D1 = D2 - D1       (SFPU)                     (b_T, packed to cb_b_full)
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
                            cb_gamma_row,
                            BroadcastDim::Row,
                            WaitPolicy::Upfront,
                            PopPolicy::AtEnd,
                            InputTileMapping::Block),
                        Dst::D0>{cols},
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
                            cb_gamma_row,
                            BroadcastDim::Row,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block),
                        Dst::D1>{cols},
                    ckl::DestReuseBinary<
                        BinaryFpuOp::Mul,
                        input(
                            cb_stats_T,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        DestReuseType::DEST_TO_SRCB,
                        Dst::D1>{0u},
                    ckl::UnaryBcast<
                        BroadcastDim::Row,
                        input(cb_beta_row, WaitPolicy::PerTile, PopPolicy::PerTile),
                        Dst::D2>{},
                    ckl::SubBinary<Dst::D2, Dst::D1, Dst::D1>{},
                    ckl::PackTile<output(cb_a_full), Dst::D0>{},
                    ckl::PackTile<output(cb_b_full), Dst::D1>{});
            }
            cb_pop_front(cb_stats_T, 2 * cols);
        }

        if (it + 1 < iters) {
            cb_wait_front(cb_a_full, cols);
            cb_pop_front(cb_a_full, cols);
            cb_wait_front(cb_b_full, cols);
            cb_pop_front(cb_b_full, cols);
        }
    }
    cb_pop_front(cb_stats_g_full, num_stats);
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def sharded_row_config(num_tiles):
    """One tile-row of `num_tiles` tiles on core (0,0) — tile t is column block t."""
    return ttnn.create_sharded_memory_config(
        shape=(TILE, num_tiles * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id, num_pages, dtype, page_bytes):
    return ttnn.CBDescriptor(
        total_size=page_bytes * num_pages,
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


def membership_tile_index(variant, tl, kg, cols, Kg):
    """Page index of E tile (channel tile tl, group tile kg) in cb_membership for a variant.
    baseline: the op's reader order tl * Kg + kg (N-major, one N per matmul call);
    batched : the matmul helper's row-major K x N order kg * cols + tl."""
    if METHODS[variant] == 0:
        return tl * Kg + kg
    return kg * cols + tl


def create_program_descriptor(
    stats_g_full, membership, gamma_row, beta_row, a_full, b_full, *, variant, cols, Kg, iters=1, zones=False
):
    method = METHODS[variant]
    ct = [
        CB_STATS_G_FULL,
        CB_MEMBERSHIP,
        CB_GAMMA_ROW,
        CB_BETA_ROW,
        CB_STATS_T,
        CB_BETA_FULL,
        CB_A_FULL,
        CB_B_FULL,
        method,
        cols,
        Kg,
        iters,
    ]
    defines = [] if zones else [("KERNEL_LIB_PERF_ZONES_OFF", "1")]
    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=ct,
        defines=defines,
        config=compute_config(),
    )
    stats_T_pages = 2 if method == 0 else 2 * cols
    beta_full_pages = 1 if method == 0 else cols
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_STATS_G_FULL, stats_g_full),
        ttnn.cb_descriptor_from_sharded_tensor(CB_MEMBERSHIP, membership),
        ttnn.cb_descriptor_from_sharded_tensor(CB_GAMMA_ROW, gamma_row),
        ttnn.cb_descriptor_from_sharded_tensor(CB_BETA_ROW, beta_row),
        _scratch_cb(CB_STATS_T, stats_T_pages, ttnn.float32, TILE_BYTES_F32),
        _scratch_cb(CB_BETA_FULL, beta_full_pages, ttnn.float32, TILE_BYTES_F32),
        ttnn.cb_descriptor_from_sharded_tensor(CB_A_FULL, a_full),
        ttnn.cb_descriptor_from_sharded_tensor(CB_B_FULL, b_full),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run_affine(device, stats_g_full, membership, gamma_row, beta_row, *, variant, cols, Kg, iters=1, zones=False):
    """Allocate a_full / b_full (fp32, cols tiles each) and run one variant. Returns (a_full, b_full)."""
    outs = []
    for _ in range(2):
        outs.append(
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([TILE, cols * TILE]), ttnn.float32, ttnn.TILE_LAYOUT, device, sharded_row_config(cols)
            )
        )
    a_full, b_full = outs
    desc = create_program_descriptor(
        stats_g_full,
        membership,
        gamma_row,
        beta_row,
        a_full,
        b_full,
        variant=variant,
        cols=cols,
        Kg=Kg,
        iters=iters,
        zones=zones,
    )
    ttnn.generic_op([stats_g_full, membership, gamma_row, beta_row, a_full, b_full], desc)
    return a_full, b_full
