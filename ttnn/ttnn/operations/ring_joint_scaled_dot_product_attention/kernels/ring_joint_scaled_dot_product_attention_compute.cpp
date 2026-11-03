// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute-core flash attention (Q-stationary, online softmax), one attention unit = Sq q tile-rows of one
// (batch, q head), walking every key segment in blocks of Sk key tiles.
//
// Per key block (skv <= Sk valid key tiles; CB credits are always nominal: Sq*Sk scores, Sk*DHt K, Sk*DVt V):
//   score_block   S = Q . K^T                      matmul_block (transpose, TileRowMajor, row stride Sk)
//   mask_block    S[:, skv-1] += key-tail mask      eltwise_chain BinaryFpu Add, Strided in place (boundary block)
//   row_max_block m_blk = rowmax(S) [max m_acc]     reduce<MAX, REDUCE_ROW> (row stride Sk) + binary_sfpu BinaryMax
//   exp_block     P = exp(s (S - m_blk)) in place   eltwise_chain BinaryFpu Sub (bcast col) -> ScaledExp, Strided
//   sum           l_blk = rowsum(P)                 reduce<SUM, REDUCE_ROW> (row stride Sk)
//   pv_block      O_blk = P . V                     matmul_block (K = Sk stride, skv inner steps)
//   rescale_block (kb > 0) a = exp(s (m_acc - m_blk)); l_acc = a l_acc + l_blk; O_acc = a O_acc + O_blk
//                 eltwise_chain BinaryFpu Mul (bcast col) -> DestReuseBinary Add, in place on the accumulators
// finalize_block: out = O_acc * (1 / l_acc)  (Recip, then Mul bcast col) -> cb_out_tiles (bf16)
//
// Deviations from the design's API table (recorded in the commit):
//  * the row sum is a reduce<SUM> over P (the design's "sum via reduce" perf lamp) instead of a second
//    L1-accumulating pack in the exp chain — l is then column-0 valid and needs no in-tile row sum at finalize;
//  * the running accumulators have fixed roles (m_acc / l_acc / O_acc) updated in place, with a per-block scratch
//    (m_blk / l_blk / O_blk), instead of a ping-pong pair — same CB count, no runtime CB-id swap;
//  * scores keep a nominal Sk row stride so every CB credit is uniform (ragged blocks narrow only the compute
//    extent: the matmul N, the reduce cols, the exp grid and the P.V inner steps).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/exp.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/minmax.hpp"
#include "ring_joint_sdpa_common.hpp"

using namespace compute_kernel_lib;

// exp(scale * x): the chain's Exp<> has no scale parameter; the SFPU exp folds the scale (SFPMULI) for free.
template <uint32_t scale_bf16>
struct ScaledExp : UnaryOp<ScaledExp<scale_bf16>, Dst::D0> {
    static ALWI void init() { exp_tile_init<false>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        exp_tile<false, true>(slot_offset, VectorMode::RC, static_cast<uint16_t>(scale_bf16));
    }
};

// P.V inner steps: only the valid key tiles of a (possibly ragged) block.
struct ValidKeyTiles {
    uint32_t n;
    ALWI uint32_t operator()(uint32_t /*block*/, uint32_t /*block_k*/) const { return n; }
};

// In-place phases on cb_scores (mask, exp) write the fronted block without moving CB credit; cycle the whole
// block's credit (unpack pop -> pack reserve/push -> unpack wait) so the next phase's unpacker reads only after the
// packer finished. The CB holds exactly one block, so the block stays on the same slots.
ALWI void inplace_fence(uint32_t cb, uint32_t n) {
    cb_pop_front(cb, n);
    cb_reserve_back(cb, n);
    cb_push_back(cb, n);
    cb_wait_front(cb, n);
}

// Largest subblock width dividing n that fits DEST (subblock height 1).
inline uint32_t subblock_w(uint32_t n) {
    uint32_t w = n < DEST_AUTO_LIMIT ? n : DEST_AUTO_LIMIT;
    while (n % w) {
        --w;
    }
    return w;
}

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = get_compile_time_arg_val(2);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(3);
    constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sum_scaler = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scores = get_compile_time_arg_val(6);
    constexpr uint32_t cb_max_blk = get_compile_time_arg_val(7);
    constexpr uint32_t cb_max_acc = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sum_blk = get_compile_time_arg_val(9);
    constexpr uint32_t cb_sum_acc = get_compile_time_arg_val(10);
    constexpr uint32_t cb_row_factor = get_compile_time_arg_val(11);
    constexpr uint32_t cb_out_blk = get_compile_time_arg_val(12);
    constexpr uint32_t cb_out_acc = get_compile_time_arg_val(13);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t Sq = get_compile_time_arg_val(15);
    constexpr uint32_t Sk = get_compile_time_arg_val(16);
    constexpr uint32_t DHt = get_compile_time_arg_val(17);
    constexpr uint32_t DVt = get_compile_time_arg_val(18);
    constexpr uint32_t n_q_t = get_compile_time_arg_val(19);
    constexpr uint32_t Lq_t = get_compile_time_arg_val(20);
    constexpr uint32_t BH = get_compile_time_arg_val(22);
    constexpr uint32_t C = get_compile_time_arg_val(23);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(26);

    const uint32_t core_idx = get_arg_val<uint32_t>(0);
    const uint32_t num_segments = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_scores);

    CircularBuffer q_buf(cb_q), k_buf(cb_k), v_buf(cb_v), scores_buf(cb_scores);
    CircularBuffer out_acc_buf(cb_out_acc), out_blk_buf(cb_out_blk);

    constexpr uint32_t U = total_units<Sq, n_q_t, Lq_t, BH, C>();
    const uint32_t my_units = units_of_core(U, C, core_idx);
    constexpr uint32_t pv_sbw_n = DVt;
    const uint32_t pv_sbw = subblock_w(pv_sbw_n);

    for (uint32_t i = 0; i < my_units; ++i) {
        uint32_t kb = 0;  // processed key blocks of this unit
        for (uint32_t s = 0; s < num_segments; ++s) {
            const uint32_t valid = get_arg_val<uint32_t>(2 + 2 * s);
            const bool seg_masked = get_arg_val<uint32_t>(3 + 2 * s) != 0;
            for (uint32_t t0 = 0; t0 < valid; t0 += Sk, ++kb) {
                const uint32_t skv = (valid - t0) < Sk ? (valid - t0) : Sk;
                const bool masked = seg_masked && (t0 + skv == valid);

                // ---- score_block: S (Sq x skv, row stride Sk) = Q . K^T ----
                k_buf.wait_front(Sk * DHt);
                {
                    const uint32_t sbw = subblock_w(skv);
                    matmul_block<
                        true,
                        false,
                        LastBlockTarget::Out,
                        OutputCBLayout::TileRowMajor,
                        matmul_config::InitMode::Short,
                        InputPolicy::WaitAndRetainOnLastBlock,
                        InputPolicy::NoWaitNoPop>(
                        q_buf,
                        k_buf,
                        scores_buf,
                        scores_buf,
                        MatmulBlockShape::of(Sq, skv / sbw, 1, sbw, DHt, 1),
                        NoPostCompute{},
                        NoPreKBlock{},
                        Sk,
                        Sk);
                }
                k_buf.pop_front(Sk * DHt);
                cb_wait_front(cb_scores, Sq * Sk);

                // ---- mask_block: last valid key tile column of the boundary block ----
                if (masked) {
                    eltwise_chain(
                        IterationShape::grid(Sq, 1),
                        BinaryFpu<
                            BinaryFpuOp::Add,
                            input(
                                cb_scores,
                                WaitPolicy::None,
                                PopPolicy::None,
                                InputTileMapping::Block,
                                DataFormatReconfig::Enabled,
                                TileAddressing::Strided),
                            input(cb_mask, BroadcastDim::None, WaitPolicy::Upfront, PopPolicy::None)>{
                            StridedTileRange{skv - 1, Sk}},
                        PackTile<output(
                            cb_scores,
                            ReservePolicy::None,
                            PushPolicy::None,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Strided)>{StridedTileRange{skv - 1, Sk}});
                    inplace_fence(cb_scores, Sq * Sk);
                }

                // ---- row_max_block ----
                reduce<
                    PoolType::MAX,
                    ReduceDim::REDUCE_ROW,
                    cb_scores,
                    cb_max_scaler,
                    cb_max_blk,
                    ReduceInputPolicy::NoWaitNoPop>(
                    ReduceInputBlockShape::of(Sq, skv), ReduceInputMemoryLayout::with_row_stride(Sk));
                if (kb > 0) {
                    binary_sfpu<
                        BinaryMax<>,
                        input(cb_max_blk),
                        input(cb_max_acc, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block),
                        output(cb_max_blk)>(IterationShape::tiles(Sq));
                }

                // ---- exp_block: P = exp(s (S - m_blk)) in place ----
                eltwise_chain(
                    IterationShape::grid(Sq, skv),
                    BinaryFpu<
                        BinaryFpuOp::Sub,
                        input(
                            cb_scores,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Strided),
                        input(
                            cb_max_blk,
                            BroadcastDim::Col,
                            WaitPolicy::Upfront,
                            PopPolicy::None,
                            InputTileMapping::Col)>{StridedTileRange{0, Sk}},
                    ScaledExp<scale_bits>{},
                    PackTile<output(
                        cb_scores,
                        ReservePolicy::None,
                        PushPolicy::None,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Strided)>{StridedTileRange{0, Sk}});
                inplace_fence(cb_scores, Sq * Sk);

                // ---- row sums of P ----
                if (kb == 0) {
                    reduce<
                        PoolType::SUM,
                        ReduceDim::REDUCE_ROW,
                        cb_scores,
                        cb_sum_scaler,
                        cb_sum_acc,
                        ReduceInputPolicy::NoWaitNoPop>(
                        ReduceInputBlockShape::of(Sq, skv), ReduceInputMemoryLayout::with_row_stride(Sk));
                } else {
                    reduce<
                        PoolType::SUM,
                        ReduceDim::REDUCE_ROW,
                        cb_scores,
                        cb_sum_scaler,
                        cb_sum_blk,
                        ReduceInputPolicy::NoWaitNoPop>(
                        ReduceInputBlockShape::of(Sq, skv), ReduceInputMemoryLayout::with_row_stride(Sk));
                }

                // ---- pv_block: O = P . V (first block straight into the accumulator) ----
                v_buf.wait_front(Sk * DVt);
                matmul_block<
                    false,
                    false,
                    LastBlockTarget::Out,
                    OutputCBLayout::TileRowMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndPopPerKBlock,
                    InputPolicy::NoWaitNoPop>(
                    scores_buf,
                    v_buf,
                    kb == 0 ? out_acc_buf : out_blk_buf,
                    kb == 0 ? out_acc_buf : out_blk_buf,
                    MatmulBlockShape::of(Sq, DVt / pv_sbw, 1, pv_sbw, Sk, 1),
                    NoPostCompute{},
                    NoPreKBlock{},
                    DVt,
                    DVt,
                    NoPostKBlock{},
                    ValidKeyTiles{skv});
                v_buf.pop_front(Sk * DVt);

                // ---- rescale_block ----
                if (kb > 0) {
                    // a = exp(s (m_acc - m_blk)); consumes m_acc
                    eltwise_chain(
                        IterationShape::tiles(Sq),
                        BinaryFpu<
                            BinaryFpuOp::Sub,
                            input(cb_max_acc),
                            input(cb_max_blk, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block)>{},
                        ScaledExp<scale_bits>{},
                        PackTile<output(cb_row_factor)>{});
                    // l_acc = a l_acc + l_blk
                    eltwise_chain(
                        IterationShape::tiles(Sq),
                        BinaryFpu<
                            BinaryFpuOp::Mul,
                            input(cb_sum_acc),
                            input(
                                cb_row_factor,
                                BroadcastDim::Col,
                                WaitPolicy::Upfront,
                                PopPolicy::None,
                                InputTileMapping::Block)>{},
                        DestReuseBinary<BinaryFpuOp::Add, input(cb_sum_blk), DestReuseType::DEST_TO_SRCA>{},
                        PackTile<output(cb_sum_acc)>{});
                    // O_acc = a O_acc + O_blk; releases a
                    eltwise_chain(
                        IterationShape::grid(Sq, DVt),
                        BinaryFpu<
                            BinaryFpuOp::Mul,
                            input(cb_out_acc),
                            input(
                                cb_row_factor,
                                BroadcastDim::Col,
                                WaitPolicy::Upfront,
                                PopPolicy::AtEnd,
                                InputTileMapping::Col)>{},
                        DestReuseBinary<BinaryFpuOp::Add, input(cb_out_blk), DestReuseType::DEST_TO_SRCA>{},
                        PackTile<output(cb_out_acc)>{});
                }
                // m_acc <- m_blk
                copy<input(cb_max_blk), output(cb_max_acc)>(IterationShape::tiles(Sq));
            }
        }

        // ---- finalize_block: out = O_acc / l_acc ----
        unary<Recip<>, input(cb_sum_acc), output(cb_row_factor)>(IterationShape::tiles(Sq));
        mul<input(cb_out_acc),
            input(cb_row_factor, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            output(cb_out_tiles)>(IterationShape::grid(Sq, DVt));
        cb_wait_front(cb_max_acc, Sq);
        cb_pop_front(cb_max_acc, Sq);
        cb_pop_front(cb_q, Sq * DHt);
    }
}
