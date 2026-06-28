// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/device_print.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
using namespace compute_kernel_lib;
constexpr uint32_t cb_q = tt::CBIndex::c_0, cb_k = tt::CBIndex::c_1, cb_v = tt::CBIndex::c_2,
                   cb_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_scaler_reduce = 4, cb_scale_factor = 5, cb_alpha = 8;
constexpr uint32_t cb_o = tt::CBIndex::c_16, cb_out = tt::CBIndex::c_17;
constexpr uint32_t cb_scores = 24, cb_scores_masked = 25, cb_max_new = 26, cb_max_old = 27;
constexpr uint32_t cb_exp_scores = 28, cb_sum_new = 29, cb_sum_old = 30, cb_o_accum = 31;

ALWI void dc(const char* n, uint32_t id) {
    DEVICE_PRINT("{}={}\n", n, (uint32_t)(get_cb_tiles_received_ptr(id)[0] - get_cb_tiles_acked_ptr(id)[0]));
}

void kernel_main() {
    constexpr uint32_t B_q_t = get_compile_time_arg_val(0), B_kv_t = get_compile_time_arg_val(1),
                       D_t = get_compile_time_arg_val(2), has_mask = get_compile_time_arg_val(3);
    constexpr uint32_t S_q_tiles = get_compile_time_arg_val(4), S_kv_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_score_tiles = B_q_t * B_kv_t, num_o_tiles = B_q_t * D_t, num_q_tiles = B_q_t * D_t;
    constexpr uint32_t num_q_blocks = (S_q_tiles + B_q_t - 1) / B_q_t,
                       num_kv_blocks = (S_kv_tiles + B_kv_t - 1) / B_kv_t;

    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_q, cb_k, cb_scores);
    mm_block_init(cb_q, cb_k, cb_scores, 1, B_kv_t, B_q_t, D_t);

    CircularBuffer q_buf(cb_q), k_buf(cb_k), v_buf(cb_v), scores_buf(cb_scores);
    CircularBuffer scores_masked_buf(cb_scores_masked), exp_scores_buf(cb_exp_scores);
    CircularBuffer o_buf(cb_o), o_accum_buf(cb_o_accum);

    constexpr uint32_t sb_h = (B_q_t < 2) ? B_q_t : 2, sb_w = (B_kv_t < 2) ? B_kv_t : 2;
    constexpr uint32_t sb_hs = (sb_h > 0) ? sb_h : 1, sb_ws = (sb_w > 0) ? sb_w : 1;
    constexpr uint32_t in0_sb = (B_q_t + sb_hs - 1) / sb_hs, in1_sb = (B_kv_t + sb_ws - 1) / sb_ws;
    constexpr auto qkt_shape = MatmulBlockShape::of(in0_sb, in1_sb, sb_hs, sb_ws, D_t, 1);
    constexpr uint32_t pv_sb_w = (D_t < 2) ? D_t : 2, pv_s = (pv_sb_w > 0) ? pv_sb_w : 1;
    constexpr uint32_t pv_in1_sb = (D_t + pv_s - 1) / pv_s;
    constexpr auto pv_shape = MatmulBlockShape::of(in0_sb, pv_in1_sb, sb_hs, pv_s, B_kv_t, 1);

    DEVICE_PRINT(
        "BOOT bqt={} bkvt={} dt={} mask={}\n", (uint32_t)B_q_t, (uint32_t)B_kv_t, (uint32_t)D_t, (uint32_t)has_mask);

    for (uint32_t qb = 0; qb < num_q_blocks; ++qb) {
        matmul_block<
            true,
            false,
            LastBlockTarget::Out,
            OutputCBLayout::SubblockMajor,
            matmul_config::InitMode::Short,
            InputPolicy::WaitAndRetainOnLastBlock,
            InputPolicy::WaitAndPopPerKBlock>(q_buf, k_buf, scores_buf, scores_buf, qkt_shape);
        DEVICE_PRINT("P1done\n");
        mul<cb_scores,
            cb_scale_factor,
            cb_scores,
            BroadcastDim::Scalar,
            InputLifecycle::Streaming,
            InputLifecycle::HeldBulk>(num_score_tiles);
        DEVICE_PRINT("P2done\n");
        if constexpr (has_mask) {
            add<cb_scores, cb_mask, cb_scores_masked>(num_score_tiles);
        } else {
            copy<cb_scores, cb_scores_masked>(num_score_tiles);
        }
        DEVICE_PRINT("P3done\n");
        reduce<
            PoolType::MAX,
            ReduceDim::REDUCE_ROW,
            cb_scores_masked,
            cb_scaler_reduce,
            cb_max_new,
            ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
        cb_wait_front(cb_scaler_reduce, 1);
        cb_pop_front(cb_scaler_reduce, 1);
        DEVICE_PRINT("P4done\n");
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
                OperandKind::Block,
                OperandKind::Block>{},
            Exp<>{},
            PackTile<cb_alpha, OutputLifecycle::Streaming>{});
        DEVICE_PRINT(
            "P5 alp={} mxo={} mxn={}\n",
            (uint32_t)(get_cb_tiles_received_ptr(cb_alpha)[0] - get_cb_tiles_acked_ptr(cb_alpha)[0]),
            (uint32_t)(get_cb_tiles_received_ptr(cb_max_old)[0] - get_cb_tiles_acked_ptr(cb_max_old)[0]),
            (uint32_t)(get_cb_tiles_received_ptr(cb_max_new)[0] - get_cb_tiles_acked_ptr(cb_max_new)[0]));
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
        DEVICE_PRINT("P6done\n");
        mul<cb_sum_old, cb_alpha, cb_sum_old>(B_q_t);
        DEVICE_PRINT(
            "P7 sno={} alp={} mxo={} mxn={}\n",
            (uint32_t)(get_cb_tiles_received_ptr(cb_sum_old)[0] - get_cb_tiles_acked_ptr(cb_sum_old)[0]),
            (uint32_t)(get_cb_tiles_received_ptr(cb_alpha)[0] - get_cb_tiles_acked_ptr(cb_alpha)[0]),
            (uint32_t)(get_cb_tiles_received_ptr(cb_max_old)[0] - get_cb_tiles_acked_ptr(cb_max_old)[0]),
            (uint32_t)(get_cb_tiles_received_ptr(cb_max_new)[0] - get_cb_tiles_acked_ptr(cb_max_new)[0]));
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
        DEVICE_PRINT("P8done\n");
        unary<Exp<>, cb_scores_masked, cb_exp_scores>(num_score_tiles);
        DEVICE_PRINT("P9done\n");
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_exp_scores,
            cb_scaler_reduce,
            cb_sum_new,
            ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
        cb_wait_front(cb_scaler_reduce, 1);
        cb_pop_front(cb_scaler_reduce, 1);
        DEVICE_PRINT(
            "P10 sno={} snn={} exp={} scr={}\n",
            (uint32_t)(get_cb_tiles_received_ptr(cb_sum_old)[0] - get_cb_tiles_acked_ptr(cb_sum_old)[0]),
            (uint32_t)(get_cb_tiles_received_ptr(cb_sum_new)[0] - get_cb_tiles_acked_ptr(cb_sum_new)[0]),
            (uint32_t)(get_cb_tiles_received_ptr(cb_exp_scores)[0] - get_cb_tiles_acked_ptr(cb_exp_scores)[0]),
            (uint32_t)(get_cb_tiles_received_ptr(cb_scaler_reduce)[0] - get_cb_tiles_acked_ptr(cb_scaler_reduce)[0]));
        DEVICE_PRINT(
            "P11 sno={} snn={}\n",
            (uint32_t)(get_cb_tiles_received_ptr(cb_sum_old)[0] - get_cb_tiles_acked_ptr(cb_sum_old)[0]),
            (uint32_t)(get_cb_tiles_received_ptr(cb_sum_new)[0] - get_cb_tiles_acked_ptr(cb_sum_new)[0]));
        add<cb_sum_old, cb_sum_new, cb_sum_old>(B_q_t);
        DEVICE_PRINT("P11done\n");
        matmul_block<
            false,
            false,
            LastBlockTarget::Out,
            OutputCBLayout::SubblockMajor,
            matmul_config::InitMode::Short,
            InputPolicy::WaitAndPopPerKBlock,
            InputPolicy::WaitAndPopPerKBlock>(exp_scores_buf, v_buf, o_accum_buf, o_accum_buf, pv_shape);
        add<cb_o, cb_o_accum, cb_o>(num_o_tiles);
        copy<cb_max_new, cb_max_old>(B_q_t);
        if (qb < num_q_blocks - 1) {
            cb_wait_front(cb_q, num_q_tiles);
            cb_pop_front(cb_q, num_q_tiles);
        }
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
    }
}
