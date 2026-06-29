// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Flash Attention compute kernel — online softmax recurrence.
// Implements the Flash Attention v2 algorithm per op_design.md:
//   For each Q-block:
//     Init m_i=-inf, l_i=0, O_i=0
//     For each KV-block:
//       QK^T matmul → scale → mask → row-max → alpha=exp(m_old-m_new) →
//       rescale O,l → sub m_new → exp → row-sum → update l → PV matmul →
//       accumulate O → update m
//     Normalize O by l_i, write to output CB

#include <cstdint>
#include <limits>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

using namespace compute_kernel_lib;

// CB indices — semantic names per design
constexpr uint32_t cb_q = tt::CBIndex::c_0;     // Q-block tiles (retained across KV-blocks)
constexpr uint32_t cb_k = tt::CBIndex::c_1;     // K-block tiles (streamed per KV-block)
constexpr uint32_t cb_v = tt::CBIndex::c_2;     // V-block tiles (streamed per KV-block)
constexpr uint32_t cb_mask = tt::CBIndex::c_3;  // attn_mask tiles (streamed per KV-block)
constexpr uint32_t cb_scale_factor = 5;         // scale value tile
constexpr uint32_t cb_alpha = 8;                // alpha = exp(m_old - m_new)
constexpr uint32_t cb_scaler_max = 6;           // reduce scaler for MAX
constexpr uint32_t cb_scaler_sum = 7;           // reduce scaler for SUM

constexpr uint32_t cb_o = tt::CBIndex::c_16;    // running output O_i
constexpr uint32_t cb_out = tt::CBIndex::c_17;  // final output (normalized)

constexpr uint32_t cb_scores = 24;         // QK^T scores
constexpr uint32_t cb_scores_masked = 25;  // scores after scale+mask
constexpr uint32_t cb_max_new = 26;        // row-max of current block
constexpr uint32_t cb_max_old = 27;        // running max m_i
constexpr uint32_t cb_exp_scores = 28;     // exp(scores - m_new)
constexpr uint32_t cb_sum_new = 29;        // row-sum of exp scores
constexpr uint32_t cb_sum_old = 30;        // running sum l_i
constexpr uint32_t cb_o_accum = 31;        // PV matmul accumulation scratch

void kernel_main() {
    constexpr uint32_t B_q_t = get_compile_time_arg_val(0);
    constexpr uint32_t B_kv_t = get_compile_time_arg_val(1);
    constexpr uint32_t D_t = get_compile_time_arg_val(2);
    constexpr uint32_t has_mask = get_compile_time_arg_val(3);
    constexpr uint32_t num_q_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t num_kv_blocks = get_compile_time_arg_val(5);

    constexpr uint32_t num_score_tiles = B_q_t * B_kv_t;
    constexpr uint32_t num_o_tiles = B_q_t * D_t;

    // Boot — hw startup with QK^T config (in0=cb_q, in1=cb_k, out=cb_scores)
    // SrcOrder::Reverse: in0→SrcB, in1→SrcA (matmul register convention)
    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_q, cb_k, cb_scores);
    mm_block_init(cb_q, cb_k, cb_scores, 1, 1, 1, D_t);

    CircularBuffer q_buf(cb_q), k_buf(cb_k), v_buf(cb_v);
    CircularBuffer scores_buf(cb_scores), scores_masked_buf(cb_scores_masked);
    CircularBuffer exp_scores_buf(cb_exp_scores), o_buf(cb_o), o_accum_buf(cb_o_accum);
    CircularBuffer out_buf(cb_out);

    constexpr auto qkt_shape = MatmulBlockShape::of(B_q_t, B_kv_t, 1, 1, D_t, 1);
    constexpr auto pv_shape = MatmulBlockShape::of(B_q_t, D_t, 1, 1, B_kv_t, 1);
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    for (uint32_t qb = 0; qb < num_q_blocks; ++qb) {
        // Phase 0: Init running state for this Q-block
        eltwise_chain(B_q_t, FillScalar<Dst::D0>{NEG_INF}, PackTile<cb_max_old, OutputLifecycle::Streaming>{});
        eltwise_chain(B_q_t, FillScalar<Dst::D0>{0.0f}, PackTile<cb_sum_old, OutputLifecycle::Streaming>{});
        eltwise_chain(num_o_tiles, FillScalar<Dst::D0>{0.0f}, PackTile<cb_o, OutputLifecycle::Streaming>{});

        for (uint32_t kvb = 0; kvb < num_kv_blocks; ++kvb) {
            // Phase 1: QK^T score matmul: S = Q @ K^T
            // in0_policy=WaitAndRetainOnLastBlock: Q retained for next KV-block
            matmul_block<true, false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::Short, InputPolicy::WaitAndRetainOnLastBlock,
                InputPolicy::WaitAndPopPerKBlock, NoPostCompute, NoPreKBlock, NoPostKBlock,
                0, NoKBlockInnerDimFn, NoIn0Source, NoIn1BaseOffset, false, NoneActivation,
                matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT>(
                q_buf, k_buf, scores_buf, scores_buf, qkt_shape);

            // Phase 2: Scale scores by scale_factor (scalar broadcast)
            mul<cb_scores, cb_scale_factor, cb_scores, BroadcastDim::Scalar,
                InputLifecycle::Streaming, InputLifecycle::HeldBulk>(num_score_tiles);

            // Phase 3a/3b: Mask add or passthrough copy
            if constexpr (has_mask) {
                add<cb_scores, cb_mask, cb_scores_masked>(EltwiseShape::grid(B_q_t, B_kv_t));
            } else {
                copy<cb_scores, cb_scores_masked>(num_score_tiles);
            }

            // Phase 4: Row-max of scores (WaitUpfrontNoPop — scores survive for phase 8)
            reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_scores_masked, cb_scaler_max, cb_max_new,
                   ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));

            // Phase 4b: Compute running max: m_new = max(m_old, m_blk)
            // The online softmax recurrence requires the RUNNING max (not just the
            // current block's max) for correct alpha and score rescaling. Without
            // this step, a fully-masked KV-block (m_blk = -inf) causes
            // alpha = exp(m_old - (-inf)) = inf, corrupting O and l.
            eltwise_chain(
                B_q_t,
                CopyTile<cb_max_old, Dst::D0, InputLifecycle::HeldBulk, CopyTileReconfig::Input, OperandKind::Block>{},
                CopyTile<
                    cb_max_new,
                    Dst::D1,
                    InputLifecycle::Streaming,
                    CopyTileReconfig::Input,
                    OperandKind::Scalar>{},
                BinaryMax<Dst::D0, Dst::D1, Dst::D0>{},
                PackTile<cb_max_new, OutputLifecycle::Streaming, PackTileReconfig::Output>{});

            // Phase 5: Compute alpha = exp(m_old - m_new)
            // cb_max_new now holds the running max = max(m_old, m_blk).
            eltwise_chain(B_q_t,
                BinaryFpu<cb_max_old, cb_max_new, BinaryFpuOp::Sub, BroadcastDim::None,
                          InputLifecycle::Bulk, InputLifecycle::HeldBulk,
                          BinaryDataFormatReconfig::Input, Dst::D0,
                          OperandKind::Block, OperandKind::Block>{},
                Exp<>{}, PackTile<cb_alpha, OutputLifecycle::Streaming>{});

            // Phase 6: Rescale O: O *= alpha (Col broadcast)
            mul<cb_o, cb_alpha, cb_o, BroadcastDim::Col,
                InputLifecycle::Streaming, InputLifecycle::HeldBulk,
                OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input,
                PackTileReconfig::Output, OperandKind::Scalar,
                OperandKind::Col>(EltwiseShape::grid(B_q_t, D_t));

            // Phase 7: Rescale l: l *= alpha (Col broadcast)
            // FIX: use EltwiseShape::col(B_q_t) to create Ht=B_q_t, Wt=1 so that
            // OperandKind::Col walks ht=0..B_q_t-1 and reads alpha[ht] for each tile.
            // Passing bare B_q_t creates EltwiseShape(Ht=1, Wt=B_q_t) which makes
            // OperandKind::Col always return ht=0, using alpha[0] for all tiles.
            mul<cb_sum_old,
                cb_alpha,
                cb_sum_old,
                BroadcastDim::Col,
                InputLifecycle::Streaming,
                InputLifecycle::HeldBulk,
                OutputLifecycle::Streaming,
                BinaryDataFormatReconfig::Input,
                PackTileReconfig::Output,
                OperandKind::Scalar,
                OperandKind::Col>(EltwiseShape::col(B_q_t));

            // Drain cb_alpha (HeldBulk — not popped by mul)
            cb_pop_front(cb_alpha, B_q_t);

            // Phase 8: Subtract m_new from scores (Col broadcast)
            sub<cb_scores_masked, cb_max_new, cb_scores_masked, BroadcastDim::Col,
                InputLifecycle::Streaming, InputLifecycle::HeldBulk,
                OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input,
                PackTileReconfig::Output, OperandKind::Scalar,
                OperandKind::Col>(EltwiseShape::grid(B_q_t, B_kv_t));

            // Phase 9: Exp of scores
            unary<Exp<>, cb_scores_masked, cb_exp_scores>(num_score_tiles);

            // Phase 10: Row-sum of exp scores
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_exp_scores, cb_scaler_sum, cb_sum_new,
                   ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));

            // Phase 11: Update l: l_i += l_blk
            add<cb_sum_old, cb_sum_new, cb_sum_old>(B_q_t);

            // Phase 12: PV matmul: P @ V → cb_o_accum
            matmul_block<false, false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::Short, InputPolicy::WaitAndPopPerKBlock,
                InputPolicy::WaitAndPopPerKBlock, NoPostCompute, NoPreKBlock, NoPostKBlock,
                0, NoKBlockInnerDimFn, NoIn0Source, NoIn1BaseOffset, false, NoneActivation,
                matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT>(
                exp_scores_buf, v_buf, o_accum_buf, o_accum_buf, pv_shape);

            // Phase 12b: O += P@V result
            add<cb_o, cb_o_accum, cb_o>(num_o_tiles);

            // Phase 13: Update m: m_i = m_new
            // Note: the online softmax recurrence uses alpha = exp(m_old - m_new) to rescale,
            // so m_i = m_new is sufficient for correctness (alpha handles the rescaling).
            copy<cb_max_new, cb_max_old>(B_q_t);
        }

        // Phase 14: Normalize O by l_i and write to output CB
        // Note: cb_o is drained by Streaming input in this mul (no separate pop needed)
        unary<Recip<>, cb_sum_old, cb_sum_old>(B_q_t);
        mul<cb_o, cb_sum_old, cb_out, BroadcastDim::Col,
            InputLifecycle::Streaming, InputLifecycle::HeldBulk,
            OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input,
            PackTileReconfig::Output, OperandKind::Scalar,
            OperandKind::Col>(EltwiseShape::grid(B_q_t, D_t));

        // Drain persistent state CBs for next Q-block
        // cb_o already drained by phase 14 mul (Streaming input pops all tiles)
        cb_pop_front(cb_sum_old, B_q_t);
        cb_pop_front(cb_max_old, B_q_t);

        // Drain cb_q so reader can push next Q-block
        if (qb < num_q_blocks - 1) {
            cb_wait_front(cb_q, num_o_tiles);
            cb_pop_front(cb_q, num_o_tiles);
        }
    }
}
