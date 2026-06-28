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

ALWI void dbg_cb(const char* n, uint32_t id) {
    uint32_t r = get_cb_tiles_received_ptr(id)[0];
    uint32_t a = get_cb_tiles_acked_ptr(id)[0];
    DEVICE_PRINT("CB {} id={} t={}\n", n, (uint32_t)id, r - a);
}
ALWI void dbg_phase(const char* p) {
    DEVICE_PRINT("== {} ==\n", p);
    dbg_cb("q",cb_q); dbg_cb("k",cb_k); dbg_cb("v",cb_v);
    dbg_cb("scr",cb_scaler_reduce); dbg_cb("scl",cb_scale_factor);
    dbg_cb("alp",cb_alpha); dbg_cb("o",cb_o);
    dbg_cb("sco",cb_scores); dbg_cb("scm",cb_scores_masked);
    dbg_cb("mxn",cb_max_new); dbg_cb("mxo",cb_max_old);
    dbg_cb("exp",cb_exp_scores); dbg_cb("snn",cb_sum_new);
    dbg_cb("sno",cb_sum_old); dbg_cb("oac",cb_o_accum);
}

void kernel_main() {
    constexpr uint32_t has_mask = get_compile_time_arg_val(0);
    constexpr uint32_t B_q_t = get_compile_time_arg_val(1);
    constexpr uint32_t B_kv_t = get_compile_time_arg_val(2);
    constexpr uint32_t D_t = get_compile_time_arg_val(3);
    constexpr uint32_t S_q_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t S_kv_tiles = get_compile_time_arg_val(5);

    constexpr uint32_t num_score_tiles = B_q_t * B_kv_t;
    constexpr uint32_t num_o_tiles = B_q_t * D_t;
    constexpr uint32_t num_q_tiles = B_q_t * D_t;
    constexpr uint32_t num_q_blocks = (S_q_tiles + B_q_t - 1) / B_q_t;
    constexpr uint32_t num_kv_blocks = (S_kv_tiles + B_kv_t - 1) / B_kv_t;

    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_q, cb_k, cb_scores);
    mm_block_init(cb_q, cb_k, cb_scores, /*transpose=*/1, /*ct_dim=*/B_kv_t, /*rt_dim=*/B_q_t, /*kt_dim=*/D_t);

    CircularBuffer q_buf(cb_q);
    CircularBuffer k_buf(cb_k);
    CircularBuffer v_buf(cb_v);
    CircularBuffer scores_buf(cb_scores);
    CircularBuffer scores_masked_buf(cb_scores_masked);
    CircularBuffer exp_scores_buf(cb_exp_scores);
    CircularBuffer o_buf(cb_o);
    CircularBuffer o_accum_buf(cb_o_accum);

    constexpr uint32_t sb_h = (B_q_t < 2) ? B_q_t : 2; constexpr uint32_t sb_h_s = sb_h > 0 ? sb_h : 1;
    constexpr uint32_t sb_w = (B_kv_t < 2) ? B_kv_t : 2; constexpr uint32_t sb_w_s = sb_w > 0 ? sb_w : 1;
    constexpr uint32_t in0_sb = (B_q_t + sb_h_s - 1) / sb_h_s;
    constexpr uint32_t in1_sb = (B_kv_t + sb_w_s - 1) / sb_w_s;
    constexpr auto qkt_shape = MatmulBlockShape::of(in0_sb, in1_sb, sb_h_s, sb_w_s, D_t, 1);
    constexpr uint32_t pv_sb_w = (D_t < 2) ? D_t : 2; constexpr uint32_t pv_sb_w_s = pv_sb_w > 0 ? pv_sb_w : 1;
    constexpr uint32_t pv_in1_sb = (D_t + pv_sb_w_s - 1) / pv_sb_w_s;
    constexpr auto pv_shape = MatmulBlockShape::of(in0_sb, pv_in1_sb, sb_h_s, pv_sb_w_s, B_kv_t, 1);

    DEVICE_PRINT("BOOT: Bqt={} Bkvt={} Dt={} mask={} nqb={} nkvb={}\n",
        (uint32_t)B_q_t, (uint32_t)B_kv_t, (uint32_t)D_t, (uint32_t)has_mask,
        (uint32_t)num_q_blocks, (uint32_t)num_kv_blocks);

    for (uint32_t qb = 0; qb < num_q_blocks; ++qb) {
        dbg_phase("P1 before QK^T");
        matmul_block<true, false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor,
            matmul_config::InitMode::Short, InputPolicy::WaitAndRetainOnLastBlock,
            InputPolicy::WaitAndPopPerKBlock>(q_buf, k_buf, scores_buf, scores_buf, qkt_shape);

        dbg_phase("P2 before scale");
        mul<cb_scores, cb_scale_factor, cb_scores, BroadcastDim::Scalar,
            InputLifecycle::Streaming, InputLifecycle::HeldBulk>(num_score_tiles);

        dbg_phase("P3 before mask");
        if constexpr (has_mask) {
            add<cb_scores, cb_mask, cb_scores_masked>(num_score_tiles);
        } else {
            copy<cb_scores, cb_scores_masked>(num_score_tiles);
        }

        dbg_phase("P4 before rowmax");
        reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_scores_masked, cb_scaler_reduce, cb_max_new,
               ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
        cb_wait_front(cb_scaler_reduce, 1);
        cb_pop_front(cb_scaler_reduce, 1);

        dbg_phase("P5 before alpha");
        eltwise_chain(B_q_t,
            BinaryFpu<cb_max_old, cb_max_new, BinaryFpuOp::Sub, BroadcastDim::None,
                      InputLifecycle::Bulk, InputLifecycle::HeldBulk,
                      BinaryDataFormatReconfig::Input, Dst::D0,
                      OperandKind::Block, OperandKind::Block>{},
            Exp<>{}, PackTile<cb_alpha, OutputLifecycle::Streaming>{});

        dbg_phase("P6 before O*=a");
        mul<cb_o, cb_alpha, cb_o, BroadcastDim::Col,
            InputLifecycle::Streaming, InputLifecycle::HeldBulk,
            OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input,
            PackTileReconfig::Output, OperandKind::Scalar, OperandKind::Col>(
            EltwiseShape::grid(B_q_t, D_t));

        dbg_phase("P7 before l*=a");
        mul<cb_sum_old, cb_alpha, cb_sum_old>(B_q_t);

        dbg_phase("P8 before S-=m");
        sub<cb_scores_masked, cb_max_new, cb_scores_masked, BroadcastDim::Col,
            InputLifecycle::Streaming, InputLifecycle::HeldBulk,
            OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input,
            PackTileReconfig::Output, OperandKind::Scalar, OperandKind::Col>(
            EltwiseShape::grid(B_q_t, B_kv_t));

        dbg_phase("P9 before P=exp");
        unary<Exp<>, cb_scores_masked, cb_exp_scores>(num_score_tiles);

        dbg_phase("P10 before rowsum");
        reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_exp_scores, cb_scaler_reduce, cb_sum_new,
               ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
        cb_wait_front(cb_scaler_reduce, 1);
        cb_pop_front(cb_scaler_reduce, 1);

        dbg_phase("P11 before l+=l_blk HANG");
        add<cb_sum_old, cb_sum_new, cb_sum_old>(B_q_t);

        dbg_phase("P12 before PV");
        matmul_block<false, false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor,
            matmul_config::InitMode::Short, InputPolicy::WaitAndPopPerKBlock,
            InputPolicy::WaitAndPopPerKBlock>(exp_scores_buf, v_buf, o_accum_buf, o_accum_buf, pv_shape);
        add<cb_o, cb_o_accum, cb_o>(num_o_tiles);

        dbg_phase("P13 before m=m_new");
        copy<cb_max_new, cb_max_old>(B_q_t);

        if (qb < num_q_blocks - 1) {
            cb_wait_front(cb_q, num_q_tiles);
            cb_pop_front(cb_q, num_q_tiles);
        }

        dbg_phase("P14 before norm");
        unary<Recip<>, cb_sum_old, cb_sum_old>(B_q_t);
        mul<cb_o, cb_sum_old, cb_out, BroadcastDim::Col,
            InputLifecycle::Streaming, InputLifecycle::HeldBulk,
            OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input,
            PackTileReconfig::Output, OperandKind::Scalar, OperandKind::Col>(
            EltwiseShape::grid(B_q_t, D_t));
    }
}
