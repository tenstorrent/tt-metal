// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <limits>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
using namespace compute_kernel_lib;
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
        eltwise_chain(B_q_t, FillScalar<Dst::D0>{NEG_INF}, PackTile<cb_max_old, OutputLifecycle::Streaming>{});
        eltwise_chain(B_q_t, FillScalar<Dst::D0>{0.0f}, PackTile<cb_sum_old, OutputLifecycle::Streaming>{});
        eltwise_chain(num_o_tiles, FillScalar<Dst::D0>{0.0f}, PackTile<cb_o, OutputLifecycle::Streaming>{});
        for (uint32_t kvb = 0; kvb < num_kv_blocks; ++kvb) {
            matmul_block<true, false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::Short, InputPolicy::WaitAndRetainOnLastBlock,
                InputPolicy::WaitAndPopPerKBlock, NoPostCompute, NoPreKBlock, NoPostKBlock,
                0, NoKBlockInnerDimFn, NoIn0Source, NoIn1BaseOffset, false, NoneActivation,
                matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT>(
                q_buf, k_buf, scores_buf, scores_buf, qkt_shape);
            mul<cb_scores, cb_scale_factor, cb_scores, BroadcastDim::Scalar,
                InputLifecycle::Streaming, InputLifecycle::HeldBulk>(num_score_tiles);
            if constexpr (has_mask) {
                add<cb_scores, cb_mask, cb_scores_masked>(num_score_tiles);
            } else {
                copy<cb_scores, cb_scores_masked>(num_score_tiles);
            }
            reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_scores_masked, cb_scaler_max, cb_max_new,
                   ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
            eltwise_chain(B_q_t,
                BinaryFpu<cb_max_old, cb_max_new, BinaryFpuOp::Sub, BroadcastDim::None,
                          InputLifecycle::Bulk, InputLifecycle::HeldBulk,
                          BinaryDataFormatReconfig::Input, Dst::D0,
                          OperandKind::Block, OperandKind::Block>{},
                Exp<>{}, PackTile<cb_alpha, OutputLifecycle::Streaming>{});
            mul<cb_o, cb_alpha, cb_o, BroadcastDim::Col,
                InputLifecycle::Streaming, InputLifecycle::HeldBulk,
                OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input,
                PackTileReconfig::Output, OperandKind::Scalar,
                OperandKind::Col>(EltwiseShape::grid(B_q_t, D_t));
            mul<cb_sum_old, cb_alpha, cb_sum_old>(B_q_t);
            sub<cb_scores_masked, cb_max_new, cb_scores_masked, BroadcastDim::Col,
                InputLifecycle::Streaming, InputLifecycle::HeldBulk,
                OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input,
                PackTileReconfig::Output, OperandKind::Scalar,
                OperandKind::Col>(EltwiseShape::grid(B_q_t, B_kv_t));
            unary<Exp<>, cb_scores_masked, cb_exp_scores>(num_score_tiles);
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_exp_scores, cb_scaler_sum, cb_sum_new,
                   ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
            add<cb_sum_old, cb_sum_new, cb_sum_old>(B_q_t);
            matmul_block<false, false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::Short, InputPolicy::WaitAndPopPerKBlock,
                InputPolicy::WaitAndPopPerKBlock, NoPostCompute, NoPreKBlock, NoPostKBlock,
                0, NoKBlockInnerDimFn, NoIn0Source, NoIn1BaseOffset, false, NoneActivation,
                matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT>(
                exp_scores_buf, v_buf, o_accum_buf, o_accum_buf, pv_shape);
            add<cb_o, cb_o_accum, cb_o>(num_o_tiles);
            copy<cb_max_new, cb_max_old>(B_q_t);
            if (qb < num_q_blocks - 1) {
                cb_wait_front(cb_q, num_o_tiles);
                cb_pop_front(cb_q, num_o_tiles);
            }
        }
        unary<Recip<>, cb_sum_old, cb_sum_old>(B_q_t);
        mul<cb_o, cb_sum_old, cb_out, BroadcastDim::Col,
            InputLifecycle::Streaming, InputLifecycle::HeldBulk,
            OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input,
            PackTileReconfig::Output, OperandKind::Scalar,
            OperandKind::Col>(EltwiseShape::grid(B_q_t, D_t));

        // Pop cb_sum_old (HeldBulk in phase 14 mul, not popped by Streaming)
        cb_pop_front(cb_sum_old, B_q_t);
    }
}
