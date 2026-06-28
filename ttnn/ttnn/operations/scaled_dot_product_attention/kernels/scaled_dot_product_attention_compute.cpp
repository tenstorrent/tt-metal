// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for scaled_dot_product_attention (Flash Attention).
//
// Flash Attention v2 recurrence with online softmax. Per (B,H) work unit:
//   For each Q-block:
//     Init m_i=-inf, l_i=0, O_i=0
//     For each KV-block:
//       1. S = Q @ K^T  (matmul_block, transpose=true)
//       2. S *= scale   (eltwise mul, scalar broadcast)
//       3a. S += mask   (eltwise add, if mask) OR
//       3b. copy S→S_masked (passthrough, if no mask)
//       4. m_blk = rowmax(S_masked)  (reduce MAX REDUCE_ROW, WaitUpfrontNoPop)
//       5. alpha = exp(m_old - m_new)  (eltwise_chain: sub + exp)
//       6. O *= alpha   (eltwise mul, Col broadcast)
//       7. l *= alpha   (eltwise mul)
//       8. S -= m_new   (eltwise sub, Col broadcast, pops S_masked)
//       9. P = exp(S)    (eltwise unary Exp)
//       10. l_blk = rowsum(P)  (reduce SUM REDUCE_ROW, WaitUpfrontNoPop)
//       11. l_i += l_blk (eltwise add)
//       12. PV = P @ V   (matmul_block, transpose=false → cb_o_accum)
//           O += PV      (eltwise add, cb_o += cb_o_accum)
//       13. m_i = m_new  (eltwise copy)
//     Normalize: O /= l_i  (recip + mul)
//     Write O to cb_out
//
// CT args: [B_q_t, B_kv_t, D_t, has_mask, num_q_blocks, num_kv_blocks]

#include <cstdint>
#include <limits>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

using namespace compute_kernel_lib;

// CB indices
constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_v = tt::CBIndex::c_2;
constexpr uint32_t cb_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_scaler_max = 6;
constexpr uint32_t cb_scaler_sum = 7;
constexpr uint32_t cb_scale_factor = 5;
constexpr uint32_t cb_alpha = 8;
constexpr uint32_t cb_o = tt::CBIndex::c_16;
constexpr uint32_t cb_out = tt::CBIndex::c_17;
constexpr uint32_t cb_scores = 24;
constexpr uint32_t cb_scores_masked = 25;
constexpr uint32_t cb_max_new = 26;
constexpr uint32_t cb_max_old = 27;
constexpr uint32_t cb_exp_scores = 28;
constexpr uint32_t cb_sum_new = 29;
constexpr uint32_t cb_sum_old = 30;
constexpr uint32_t cb_o_accum = 31;

void kernel_main() {
    constexpr uint32_t B_q_t = get_compile_time_arg_val(0);
    constexpr uint32_t B_kv_t = get_compile_time_arg_val(1);
    constexpr uint32_t D_t = get_compile_time_arg_val(2);
    constexpr uint32_t has_mask = get_compile_time_arg_val(3);
    constexpr uint32_t num_q_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t num_kv_blocks = get_compile_time_arg_val(5);

    constexpr uint32_t num_score_tiles = B_q_t * B_kv_t;
    constexpr uint32_t num_o_tiles = B_q_t * D_t;

    // --- Boot: engine-wide init + matmul init for QK^T ---
    // Use 1x1 subblocks for correct row-major tile ordering.
    // With 1x1 subblocks + SubblockMajor, tiles are packed in row-major order,
    // which is compatible with the reduce helper's row-major tile access pattern.
    // QK^T: M=B_q_t, N=B_kv_t, K=D_t
    // PV:   M=B_q_t, N=D_t,    K=B_kv_t
    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_q, cb_k, cb_scores);
    mm_block_init(
        cb_q,
        cb_k,
        cb_scores,
        /*transpose=*/1,
        /*ct_dim=*/1,
        /*rt_dim=*/1,
        /*kt_dim=*/D_t);

    CircularBuffer q_buf(cb_q);
    CircularBuffer k_buf(cb_k);
    CircularBuffer v_buf(cb_v);
    CircularBuffer scores_buf(cb_scores);
    CircularBuffer scores_masked_buf(cb_scores_masked);
    CircularBuffer exp_scores_buf(cb_exp_scores);
    CircularBuffer o_buf(cb_o);
    CircularBuffer o_accum_buf(cb_o_accum);
    CircularBuffer out_buf(cb_out);

    constexpr auto qkt_shape = MatmulBlockShape::of(B_q_t, B_kv_t, 1, 1, D_t, 1);
    constexpr auto pv_shape = MatmulBlockShape::of(B_q_t, D_t, 1, 1, B_kv_t, 1);

    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    for (uint32_t qb = 0; qb < num_q_blocks; ++qb) {
        // --- Phase 0: Init running state ---
        // Fill m_old with -inf, l_old with 0, O with 0 for this Q-block.
        eltwise_chain(B_q_t, FillScalar<Dst::D0>{NEG_INF}, PackTile<cb_max_old, OutputLifecycle::Streaming>{});
        eltwise_chain(B_q_t, FillScalar<Dst::D0>{0.0f}, PackTile<cb_sum_old, OutputLifecycle::Streaming>{});
        eltwise_chain(num_o_tiles, FillScalar<Dst::D0>{0.0f}, PackTile<cb_o, OutputLifecycle::Streaming>{});

        for (uint32_t kvb = 0; kvb < num_kv_blocks; ++kvb) {
            // --- Phase 1: QK^T score matmul ---
            // Q retained (WaitAndRetainOnLastBlock with num_k_blocks=1 → no pop).
            // K consumed (WaitAndPopPerKBlock).
            matmul_block<
                /*transpose=*/true,
                /*packer_l1_acc=*/false,
                /*last_block_target=*/LastBlockTarget::Out,
                /*tile_order=*/OutputCBLayout::SubblockMajor,
                /*init_mode=*/matmul_config::InitMode::Short,
                /*in0_policy=*/InputPolicy::WaitAndRetainOnLastBlock,
                /*in1_policy=*/InputPolicy::WaitAndPopPerKBlock,
                /*reconfig=*/matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT>(
                q_buf, k_buf, scores_buf, scores_buf, qkt_shape);

            // --- Phase 2: Scale scores (scalar broadcast, in-place) ---
            mul<cb_scores,
                cb_scale_factor,
                cb_scores,
                BroadcastDim::Scalar,
                InputLifecycle::Streaming,
                InputLifecycle::HeldBulk>(num_score_tiles);

            // --- Phase 3: Mask add or passthrough ---
            if constexpr (has_mask) {
                add<cb_scores, cb_mask, cb_scores_masked>(num_score_tiles);
            } else {
                copy<cb_scores, cb_scores_masked>(num_score_tiles);
            }

            // --- Phase 4: Row-max (WaitUpfrontNoPop — leaves scores_masked for phase 8) ---
            reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_ROW,
                cb_scores_masked,
                cb_scaler_max,
                cb_max_new,
                ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));

            // --- Phase 5: alpha = exp(m_old - m_new) ---
            // m_old: B_q_t tiles, Bulk (wait all, pop all at end)
            // m_new: B_q_t tiles, HeldBulk (wait all, no pop — needed for phases 8, 13)
            eltwise_chain(
                B_q_t,
                BinaryFpu<
                    cb_max_old,
                    cb_max_new,
                    BinaryFpuOp::Sub,
                    BroadcastDim::None,
                    InputLifecycle::Bulk,
                    InputLifecycle::HeldBulk,
                    BinaryDataFormatReconfig::Input,
                    Dst::D0,
                    OperandKind::Scalar,
                    OperandKind::Scalar>{},
                Exp<>{},
                PackTile<cb_alpha, OutputLifecycle::Streaming>{});

            // --- Phase 6: O *= alpha (Col broadcast) ---
            // O: num_o_tiles (B_q_t * D_t), Scalar OperandKind (in-place streaming)
            // alpha: B_q_t tiles, Col OperandKind, HeldBulk (wait all, no pop — needed for phase 7)
            mul<cb_o,
                cb_alpha,
                cb_o,
                BroadcastDim::Col,
                InputLifecycle::Streaming,
                InputLifecycle::HeldBulk,
                OutputLifecycle::Streaming,
                BinaryDataFormatReconfig::Input,
                PackTileReconfig::Output,
                OperandKind::Scalar,
                OperandKind::Col>(EltwiseShape::grid(B_q_t, D_t));

            // --- Phase 7: l *= alpha (1D, in-place) ---
            // alpha: B_q_t tiles, Streaming (wait+pop per tile — consumed here)
            mul<cb_sum_old, cb_alpha, cb_sum_old>(B_q_t);

            // --- Phase 8: S -= m_new (Col broadcast) ---
            // scores_masked: num_score_tiles, Streaming (wait+pop — consumes scores_masked)
            // m_new: B_q_t tiles, HeldBulk (wait all, no pop — needed for phase 13)
            sub<cb_scores_masked,
                cb_max_new,
                cb_scores_masked,
                BroadcastDim::Col,
                InputLifecycle::Streaming,
                InputLifecycle::HeldBulk,
                OutputLifecycle::Streaming,
                BinaryDataFormatReconfig::Input,
                PackTileReconfig::Output,
                OperandKind::Scalar,
                OperandKind::Col>(EltwiseShape::grid(B_q_t, B_kv_t));

            // --- Phase 9: P = exp(S - m_new) ---
            unary<Exp<>, cb_scores_masked, cb_exp_scores>(num_score_tiles);

            // --- Phase 10: l_blk = rowsum(P) (WaitUpfrontNoPop) ---
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_exp_scores,
                cb_scaler_sum,
                cb_sum_new,
                ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));

            // --- Phase 11: l_i += l_blk ---
            add<cb_sum_old, cb_sum_new, cb_sum_old>(B_q_t);

            // --- Phase 12: PV = P @ V, then O += PV ---
            // P: cb_exp_scores (WaitUpfrontNoPop left tiles at front, need to pop)
            // V: cb_v (pushed by reader)
            // Output to cb_o_accum, then add into cb_o
            matmul_block<
                /*transpose=*/false,
                /*packer_l1_acc=*/false,
                /*last_block_target=*/LastBlockTarget::Out,
                /*tile_order=*/OutputCBLayout::SubblockMajor,
                /*init_mode=*/matmul_config::InitMode::Short,
                /*in0_policy=*/InputPolicy::WaitAndPopPerKBlock,
                /*in1_policy=*/InputPolicy::WaitAndPopPerKBlock,
                /*reconfig=*/matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT>(
                exp_scores_buf, v_buf, o_accum_buf, o_accum_buf, pv_shape);

            // O += PV (in-place)
            add<cb_o, cb_o_accum, cb_o>(num_o_tiles);

            // --- Phase 13: m_i = m_new ---
            // m_new: B_q_t tiles (HeldBulk — still at front, not popped)
            // Output to m_old (which was popped by Bulk in phase 5)
            copy<cb_max_new, cb_max_old>(B_q_t);

            // Pop Q tiles after last KV-block if more Q-blocks remain
            if (qb < num_q_blocks - 1) {
                cb_wait_front(cb_q, num_o_tiles);
                cb_pop_front(cb_q, num_o_tiles);
            }
        }

        // --- Phase 14: Normalize O /= l_i and write to cb_out ---
        // recip(l_i) in-place on cb_sum_old
        unary<Recip<>, cb_sum_old, cb_sum_old>(B_q_t);

        // O *= recip(l_i) → cb_out (Col broadcast)
        mul<cb_o,
            cb_sum_old,
            cb_out,
            BroadcastDim::Col,
            InputLifecycle::Streaming,
            InputLifecycle::HeldBulk,
            OutputLifecycle::Streaming,
            BinaryDataFormatReconfig::Input,
            PackTileReconfig::Output,
            OperandKind::Scalar,
            OperandKind::Col>(EltwiseShape::grid(B_q_t, D_t));
    }
}
