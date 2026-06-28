// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
// CT args: [has_mask, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles]
// RT args:  [num_work_units]
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/device_print.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
using namespace compute_kernel_lib;
constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_v = tt::CBIndex::c_2;
constexpr uint32_t cb_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_scaler_reduce = 4;
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
    constexpr uint32_t has_mask = get_compile_time_arg_val(0);
    constexpr uint32_t B_q_t = get_compile_time_arg_val(1);
    constexpr uint32_t B_kv_t = get_compile_time_arg_val(2);
    constexpr uint32_t D_t = get_compile_time_arg_val(3);
    constexpr uint32_t S_q_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t S_kv_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_score_tiles = B_q_t * B_kv_t;
    constexpr uint32_t num_o_tiles = B_q_t * D_t;
    const uint32_t num_q_blocks = (S_q_tiles + B_q_t - 1) / B_q_t;
    const uint32_t num_kv_blocks = (S_kv_tiles + B_kv_t - 1) / B_kv_t;
    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_q, cb_k, cb_scores);
    constexpr uint32_t sb_h = (B_q_t < 2) ? B_q_t : 2;
    constexpr uint32_t sb_w = (B_kv_t < 2) ? B_kv_t : 2;
    constexpr uint32_t in0_sb = (B_q_t + sb_h - 1) / sb_h;
    constexpr uint32_t in1_sb = (B_kv_t + sb_w - 1) / sb_w;
    mm_block_init(cb_q, cb_k, cb_scores, 1, sb_w, sb_h, D_t);
    CircularBuffer q_buf(cb_q);
    CircularBuffer k_buf(cb_k);
    CircularBuffer v_buf(cb_v);
    CircularBuffer scores_buf(cb_scores);
    CircularBuffer scores_masked_buf(cb_scores_masked);
    CircularBuffer exp_scores_buf(cb_exp_scores);
    CircularBuffer o_buf(cb_o);
    CircularBuffer o_accum_buf(cb_o_accum);
    constexpr auto qkt_shape = MatmulBlockShape::of(in0_sb, in1_sb, sb_h, sb_w, D_t, 1);
    constexpr uint32_t pv_sb_w = (D_t < 2) ? D_t : 2;
    constexpr uint32_t pv_in1_sb = (D_t + pv_sb_w - 1) / pv_sb_w;
    constexpr auto pv_shape = MatmulBlockShape::of(in0_sb, pv_in1_sb, sb_h, pv_sb_w, B_kv_t, 1);
    uint32_t num_work_units = get_arg_val<uint32_t>(0);
    for (uint32_t wu = 0; wu < num_work_units; ++wu) {
        DEVICE_PRINT("COMPUTE: wu={} start\n", wu);
        for (uint32_t qb = 0; qb < num_q_blocks; ++qb) {
            DEVICE_PRINT("COMPUTE: qb={} waiting for max_old\n", qb);
            cb_wait_front(cb_max_old, B_q_t);
            DEVICE_PRINT("COMPUTE: qb={} waiting for sum_old\n", qb);
            cb_wait_front(cb_sum_old, B_q_t);
            DEVICE_PRINT("COMPUTE: qb={} waiting for o\n", qb);
            cb_wait_front(cb_o, num_o_tiles);
            DEVICE_PRINT("COMPUTE: qb={} waiting for q\n", qb);
            cb_wait_front(cb_q, num_o_tiles);
            DEVICE_PRINT("COMPUTE: qb={} Phase 0 done, entering KV loop\n", qb);
            for (uint32_t kvb = 0; kvb < num_kv_blocks; ++kvb) {
                DEVICE_PRINT("COMPUTE: qb={} kvb={} before QK^T matmul\n", qb, kvb);
                // Phase 1: QK^T
                matmul_block<
                    true,
                    false,
                    LastBlockTarget::Out,
                    OutputCBLayout::SubblockMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndRetainOnLastBlock,
                    InputPolicy::WaitAndPopPerKBlock>(q_buf, k_buf, scores_buf, scores_buf, qkt_shape);
                DEVICE_PRINT("COMPUTE: qb={} kvb={} after QK^T matmul\n", qb, kvb);
                // Phase 2: Scale
                mul<cb_scores,
                    cb_scale_factor,
                    cb_scores,
                    BroadcastDim::Scalar,
                    InputLifecycle::Streaming,
                    InputLifecycle::HeldBulk>(num_score_tiles);
                // Phase 3: Mask or passthrough
                if constexpr (has_mask) {
                    add<cb_scores, cb_mask, cb_scores_masked>(num_score_tiles);
                } else {
                    copy<cb_scores, cb_scores_masked>(num_score_tiles);
                }
                // Phase 4: Row-max
                reduce<
                    PoolType::MAX,
                    ReduceDim::REDUCE_ROW,
                    cb_scores_masked,
                    cb_scaler_reduce,
                    cb_max_new,
                    ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
                cb_wait_front(cb_scaler_reduce, 1);
                cb_pop_front(cb_scaler_reduce, 1);
                // Phase 5: alpha = exp(m_old - m_new) — Streaming+Scalar (not Bulk+Block)
                eltwise_chain(
                    B_q_t,
                    BinaryFpu<
                        cb_max_old,
                        cb_max_new,
                        BinaryFpuOp::Sub,
                        BroadcastDim::None,
                        InputLifecycle::Streaming,
                        InputLifecycle::HeldBulk,
                        BinaryDataFormatReconfig::Input,
                        Dst::D0,
                        OperandKind::Scalar,
                        OperandKind::Scalar>{},
                    Exp<>{},
                    PackTile<cb_alpha, OutputLifecycle::Streaming>{});
                // Phase 6: O *= alpha (Col broadcast) — Streaming+Scalar
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
                // Phase 7: l *= alpha
                mul<cb_sum_old, cb_alpha, cb_sum_old>(B_q_t);
                // Phase 8: S -= m_new (Col broadcast) — Streaming+Scalar
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
                // Phase 9: P = exp(S)
                unary<Exp<>, cb_scores_masked, cb_exp_scores>(num_score_tiles);
                // Phase 10: rowsum(P) — WaitUpfrontNoPop (leaves exp_scores for PV matmul)
                reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_ROW,
                    cb_exp_scores,
                    cb_scaler_reduce,
                    cb_sum_new,
                    ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
                cb_wait_front(cb_scaler_reduce, 1);
                cb_pop_front(cb_scaler_reduce, 1);
                // Phase 11: l_i += l_blk (output to cb_o_accum to avoid in-place deadlock, then copy back)
                add<cb_sum_old, cb_sum_new, cb_o_accum>(B_q_t);
                copy<cb_o_accum, cb_sum_old>(B_q_t);
                // Phase 12: PV = P @ V -> cb_o_accum, O += PV
                matmul_block<
                    false,
                    false,
                    LastBlockTarget::Out,
                    OutputCBLayout::SubblockMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndPopPerKBlock,
                    InputPolicy::WaitAndPopPerKBlock>(exp_scores_buf, v_buf, o_accum_buf, o_accum_buf, pv_shape);
                add<cb_o, cb_o_accum, cb_o>(num_o_tiles);
                // Phase 13: m_i = m_new
                copy<cb_max_new, cb_max_old>(B_q_t);
            }
            // Phase 14: Normalize O /= l_i
            unary<Recip<>, cb_sum_old, cb_sum_old>(B_q_t);
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
            cb_pop_front(cb_max_old, B_q_t);
            cb_pop_front(cb_sum_old, B_q_t);
            cb_pop_front(cb_q, num_o_tiles);
        }
    }
}
