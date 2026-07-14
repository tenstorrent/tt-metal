// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for rms_norm.
//
//   RMSNorm(x) = x / sqrt(mean(x^2, dim=-1) + epsilon) * gamma
//
// Parameterized row-parallel streaming reduction. Per assigned tile-row:
//   Pass 1 (stream W in W_BLOCK_TILES chunks):
//     (rm) tilize -> square -> accumulate_reduce_block<SUM,ROW>  =>  Sum(x^2)/W
//   Finalize (1 tile, in place):
//     transform_in_place: add epsilon (SFPU) then rsqrt (SFPU)    =>  1/rms
//   Pass 2 (stream W again, re-reading x):
//     (rm) tilize -> [ (gamma) tilize gamma ] -> mul<Col> x*(1/rms)
//       -> [ mul<Row> *gamma ] -> [ (rm) untilize ]
//
// The Sum(x^2) accumulator (cb_sumsq) is a single-thread compute scratch: the
// reduce writes it, transform_in_place rewrites it in place, the pass-2 mul
// reads it held (never popped mid-row), and it is popped once at row end.
//
// Every phase is a kernel_lib helper. The only raw LLK is the epsilon+rsqrt
// finalizer inside transform_in_place's documented lambda hook (its intended
// use: "a chain like mul_unary_tile, add_unary_tile, rsqrt_tile").
//
// HELPER NOTE: the streaming-reduce wrapper accumulate_reduce_block() is stale
// against the current reduce() (it forwards CB ids as RUNTIME args, but reduce()
// now takes them as TEMPLATE args, per reduce_helpers_compute.hpp:482 and its
// examples at :43-51) — a fresh compile of the wrapper fails. So the per-block
// accumulating reduce is expressed directly on the lower-level reduce() helper
// (same behaviour the wrapper documents: Accumulate::at(cb, b) each block, and
// the partial scaler routed only to the last block). transform_in_place, which
// does not touch reduce(), is used as-is.
//
// REDUCE DATAPATH (Refinement 1 — fp32 Σx² scale bug; Refinement 2a — bf16 sibling).
// The Σx² reduce runs on one of two reduce() datapaths, picked at compile time via
// USE_ACC_VIA_ADD (host = float(fp32|bf16) && tile-aligned):
//   * tile-aligned float (USE_ACC_VIA_ADD): ReduceAlgorithm::AccumulateViaAdd.
//     The cb_sumsq accumulator holds the RAW element-wise Σx² tile, folded natively
//     per block with add_tiles and reduced ONCE (sfpu_reduce) on the last block. SUM
//     has no scaler tile, so 1/W (the mean) is applied on the last block via the
//     post_reduce_op hook. This removes the per-block reduced-partial reload of the
//     ReduceTile path, whose roundtrip undercounted mean(x²) linearly in W (fp32:
//     got/true ≈ 1 + 2.5e-6·W) / catastrophically at very wide W (bf16 W=32768:
//     rel-RMS 0.40). cb_sumsq holds mean(x²) after the loop — identical contract to
//     the ReduceTile path, so the finalize + pass 2 are shared.
//     R2a: cb_sumsq is fp32 for BOTH fp32 and bf16 input (host-forced) so the raw
//     running sum never truncates. For bf16 input the reduce helper folds the bf16
//     cb_xsq into the fp32 accumulator natively (SRCB/SRCA reconfig around the
//     acc-add; see reduce_helpers_compute.inl). A bf16 accumulator (the pre-R2a
//     state) hit the W=32768 cliff; merely forcing cb_sumsq fp32 on the *ReduceTile*
//     path was measured a net regression (R2 null result) — the fix is the fp32
//     accumulator ON this AccumulateViaAdd datapath, which carries no ∝W bias.
//   * bf8b (any alignment) and fp32/bf16 non-tile-aligned: the unchanged ReduceTile
//     matmul-with-ones path + partial scaler on the last block's last tile.
//     AccumulateViaAdd's cross-call accumulate cannot express the masked partial
//     tile, and bf8b already passes on ReduceTile (R2) — both keep ReduceTile.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

// CROSS-CORE COMBINE (Refinement 5 — WIDTH_SHARDED / BLOCK_SHARDED). When
// IS_CROSS_CORE is set the hidden W is split across a reduction GROUP of cores,
// so pass 1 produces this core's PARTIAL mean contribution (local Σx²/W_global)
// into cb_partial instead of finalizing locally. The dataflow reader gathers the
// group's partials into cb_gather on the group's ROOT; the root folds them with
// the SAME reduce<AccumulateViaAdd> helper (element-wise sum of the col-0-valid
// partials, then the single column-collapse = global mean(x²)), rsqrt-finalizes
// into cb_rms_src, and the reader broadcasts the result into cb_sumsq on every
// group member (loopback fills the root's own). Non-root cores skip the fold —
// their cb_sumsq is filled by the reader's ReceiverPipe. Pass 2 (normalize) is
// shared, reading the per-row 1/rms from cb_sumsq exactly as the local path.
namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_gamma_rm = 3;
constexpr uint32_t cb_gamma_tiles = 4;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
constexpr uint32_t cb_xsq = 24;
constexpr uint32_t cb_sumsq = 25;
constexpr uint32_t cb_norm = 26;
constexpr uint32_t cb_partial = 27;  // R5: this core's local partial Σx²/W_global (cross-core)
constexpr uint32_t cb_gather = 28;   // R5: group partials gathered on the root
constexpr uint32_t cb_rms_src = 29;  // R5: root's finalized 1/rms, source of the broadcast (ONE clean push)
constexpr uint32_t cb_combine = 30;  // R5: root's combine accumulator (churns per fold block)
constexpr uint32_t TILE_H = 32;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(0);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(1);
    constexpr uint32_t HAS_PARTIAL_W = get_compile_time_arg_val(2);
    constexpr uint32_t origin_H = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_image = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t W_BLOCK_TILES = get_compile_time_arg_val(6);
    constexpr uint32_t num_w_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t RECIP_W_BITS = get_compile_time_arg_val(8);  // 1/W float bits (mean scaler)
    // Selects the AccumulateViaAdd reduce datapath (R1 fp32; R2a bf16). Host folds
    // in !has_partial_w, so the gate is just this flag. bf8b + non-tile-aligned use
    // ReduceTile.
    constexpr uint32_t USE_ACC_VIA_ADD = get_compile_time_arg_val(9);
    // Gamma layout leg: RM gamma is tilized here; TILE gamma arrives already
    // tiled from the reader (skip the gamma-tilize step). Independent of the
    // input layout (IS_ROW_MAJOR).
    constexpr uint32_t GAMMA_IS_ROW_MAJOR = get_compile_time_arg_val(10);
    // R5 cross-core: split-W scheme-change. When set, pass 1 writes a PARTIAL
    // (cb_partial) that the root folds across the group; GROUP_SIZE is the group's
    // core count (only the root's fold loop reads it).
    constexpr uint32_t IS_CROSS_CORE = get_compile_time_arg_val(11);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(12);
    (void)Wt;
    // Pass-1 reduce target: the cross-core path accumulates into cb_partial (later
    // gathered + folded); the local path accumulates straight into cb_sumsq.
    constexpr uint32_t cb_reduce_out = (IS_CROSS_CORE != 0) ? cb_partial : cb_sumsq;

    uint32_t num_tile_rows = get_arg_val<uint32_t>(0);
    uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    uint32_t eps_bits = get_arg_val<uint32_t>(2);
    uint32_t is_root = get_arg_val<uint32_t>(3);  // R5: group root flag (cross-core only)

    compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles);

    constexpr auto reduce_shape = ckl::ReduceInputBlockShape::row(W_BLOCK_TILES);
    constexpr auto wshape = ckl::EltwiseShape::tiles(W_BLOCK_TILES);

    for (uint32_t t = 0; t < num_tile_rows; ++t) {
        // Every tile-row is processed as a full 32-row tile in both regimes
        // (the RM reader zero-pads H-padding rows; TILE gets ttnn's zero
        // padding). Padding rows reduce to 0 and are dropped by the writer.

        // ---------- Pass 1: mean(x^2) -> cb_sumsq ----------
        for (uint32_t b = 0; b < num_w_blocks; ++b) {
            if constexpr (IS_ROW_MAJOR) {
                ckl::tilize<W_BLOCK_TILES, cb_input_rm, cb_input_tiles>(1, TILE_H);
            }
            ckl::square<cb_input_tiles, cb_xsq>(wshape);
            // Accumulating SUM reduce over W: fresh at b==0, reload+add after.
            const bool is_last = (b + 1 == num_w_blocks);
            if constexpr (USE_ACC_VIA_ADD) {
                // Accurate datapath (R1 fp32; R2a bf16): fold the RAW Σx² tile in the
                // fp32 accumulator (add_tiles), reduce once on the last block, and apply
                // 1/W (the mean) via the last-block post_reduce_op — SUM carries no scaler
                // tile. RECIP_W_BITS is a compile-time constant so the lambda needs no
                // capture. cb_sumsq is fp32 here (set host-side) so the raw running sum
                // does not truncate; the reduce helper folds the bf16 cb_xsq into it
                // natively (SRCB/SRCA reconfig around the acc-add).
                const auto acc =
                    is_last ? ckl::Accumulate::at_last(cb_reduce_out, b) : ckl::Accumulate::at(cb_reduce_out, b);
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_xsq,
                    cb_scaler,
                    cb_reduce_out,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ckl::ReduceAlgorithm::AccumulateViaAdd>(
                    reduce_shape, ckl::ReduceInputMemoryLayout::contiguous(), acc, [](uint32_t dst) {
                        binop_with_scalar_tile_init();
                        mul_unary_tile(dst, RECIP_W_BITS);  // × 1/W_global → (partial) mean(x²)
                    });
            } else {
                // ReduceTile matmul-with-ones + partial scaler on the last block's last tile
                // (the partial scaler zeros the non-tile-aligned W tail; the scaler carries 1/W).
                ckl::ReducePartialScaler part = (HAS_PARTIAL_W && is_last) ? ckl::ReducePartialScaler::last_tile_at(1)
                                                                           : ckl::ReducePartialScaler::none();
                ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, cb_xsq, cb_scaler, cb_reduce_out>(
                    reduce_shape,
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(cb_reduce_out, b),
                    ckl::NoOp{},
                    part);
            }
        }

        // ---------- Finalize: 1/sqrt(mean + eps) → cb_sumsq (held across pass 2) ----------
        if constexpr (IS_CROSS_CORE) {
            // Cross-core combine (root only): fold the group's GROUP_SIZE partial tiles
            // (cb_gather) into cb_rms_src with reduce<AccumulateViaAdd> — add_tiles
            // element-wise sum, then one REDUCE_ROW column-collapse — then rsqrt-finalize.
            // The reader broadcasts the result to the group's cb_sumsq. Non-root cores:
            // their cb_sumsq is filled by the reader's ReceiverPipe; nothing to do here.
            //
            // Correctness relies on pass 1 producing col-0-ONLY partials (ReduceTile
            // matmul-with-ones zeros the other columns), so the combine's column-collapse
            // is idempotent (sums 0s). USE_ACC_VIA_ADD is 0 on the cross-core path for
            // exactly this reason (AccumulateViaAdd's sfpu finalize leaves the other
            // columns unspecified, which the collapse would fold into col 0 — a scale bug).
            //
            // The fold accumulates into cb_combine (which the reduce push/pops PER block),
            // then rsqrt-finalizes in place, then a SINGLE `copy` emits cb_rms_src with
            // exactly ONE push. cb_rms_src must have one clean lifetime push so the reader's
            // cb_wait_front(cb_rms_src, 1) cannot race and grab a mid-fold intermediate
            // (that bug shipped a partial as the broadcast value — DEVICE_PRINT-confirmed).
            if (is_root) {
                constexpr auto combine_shape = ckl::ReduceInputBlockShape::row(1);
                for (uint32_t g = 0; g < GROUP_SIZE; ++g) {
                    const bool g_last = (g + 1 == GROUP_SIZE);
                    const auto gacc =
                        g_last ? ckl::Accumulate::at_last(cb_combine, g) : ckl::Accumulate::at(cb_combine, g);
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_gather,
                        cb_scaler,
                        cb_combine,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        combine_shape, ckl::ReduceInputMemoryLayout::contiguous(), gacc, ckl::NoOp{});
                }
                ckl::transform_in_place(cb_combine, [eps_bits](uint32_t dst) {
                    binop_with_scalar_tile_init();
                    add_unary_tile(dst, eps_bits);
                    rsqrt_tile_init();
                    rsqrt_tile(dst);
                });
                ckl::copy<cb_combine, cb_rms_src>(ckl::EltwiseShape::single());
            }
        } else {
            ckl::transform_in_place(cb_sumsq, [eps_bits](uint32_t dst) {
                binop_with_scalar_tile_init();
                add_unary_tile(dst, eps_bits);
                rsqrt_tile_init();
                rsqrt_tile(dst);
            });
        }

        // ---------- Pass 2: x * (1/rms) * gamma ----------
        for (uint32_t b = 0; b < num_w_blocks; ++b) {
            if constexpr (IS_ROW_MAJOR) {
                ckl::tilize<W_BLOCK_TILES, cb_input_rm, cb_input_tiles>(1, TILE_H);
            }
            if constexpr (HAS_GAMMA) {
                if constexpr (GAMMA_IS_ROW_MAJOR) {
                    // RM gamma: one stick -> a row-0-valid tile (asymmetric tilize, 1 input page).
                    ckl::tilize<W_BLOCK_TILES, cb_gamma_rm, cb_gamma_tiles>(1, 1);
                }
                // TILE gamma: reader already pushed row-0-valid tiles into cb_gamma_tiles.
                // x * (1/rms): B (cb_sumsq) is a per-row scalar valid in col 0, held across the whole
                // row (HeldBulk, never popped here), broadcast across columns (Col).
                ckl::mul<
                    cb_input_tiles,
                    cb_sumsq,
                    cb_norm,
                    ckl::BroadcastDim::Col,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Scalar,
                    ckl::OperandKind::Scalar>(wshape);
                // * gamma: gamma weight is valid in row 0, broadcast down all rows (Row).
                ckl::mul<cb_norm, cb_gamma_tiles, cb_output_tiles, ckl::BroadcastDim::Row>(wshape);
            } else {
                ckl::mul<
                    cb_input_tiles,
                    cb_sumsq,
                    cb_output_tiles,
                    ckl::BroadcastDim::Col,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Scalar,
                    ckl::OperandKind::Scalar>(wshape);
            }
            if constexpr (IS_ROW_MAJOR) {
                ckl::untilize<W_BLOCK_TILES, cb_output_tiles, cb_output_rm>(1);
            }
        }

        // Release the per-row 1/rms held across pass 2.
        cb_pop_front(cb_sumsq, 1);
    }

    cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
}
