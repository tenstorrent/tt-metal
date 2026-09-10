// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel/compute/dest_format_helpers.hpp"

// CSA pool on 1x32 ROW_MAJOR faces. This core owns a Dh-slice (Ca); the reader has
// copied the matching Cb shard. For each user and local width-tile:
//   logits = [prev_gate_ca + bias_ca | win_gate_cb + bias_cb]     (2*cr faces)
//   out    = sum(softmax(logits) * [prev_kv_ca | win_kv_cb])
namespace {

// out_of_order_output is required: the default pack_tile ignores out_idx and just
// appends sequentially, and the logits below are produced interleaved (Ca then Cb).
void pack_face(uint32_t cb_out, uint32_t out_idx = 0) {
#if defined FP32_DEST_ACC_EN
    pack_reconfig_data_format(cb_out);
#endif
    pack_tile<true>(0, cb_out, out_idx);
}

void add_faces(uint32_t cb_a, uint32_t ia, uint32_t cb_b, uint32_t ib, uint32_t cb_out, uint32_t out_idx) {
    add_init(cb_a, cb_b);
    tile_regs_acquire();
    add_tiles(cb_a, cb_b, ia, ib, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_face(cb_out, out_idx);
    tile_regs_release();
}

void copy_face(uint32_t cb_in, uint32_t in_idx, uint32_t cb_out) {
    copy_tile_init_with_dt(cb_in);
    tile_regs_acquire();
    copy_tile(cb_in, in_idx, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_face(cb_out);
    tile_regs_release();
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_prev_kv = get_compile_time_arg_val(0);
    constexpr uint32_t cb_prev_gate = get_compile_time_arg_val(1);
    constexpr uint32_t cb_bias_ca = get_compile_time_arg_val(2);
    constexpr uint32_t cb_win_kv = get_compile_time_arg_val(3);
    constexpr uint32_t cb_win_gate = get_compile_time_arg_val(4);
    constexpr uint32_t cb_bias_cb = get_compile_time_arg_val(5);
    constexpr uint32_t cb_logits = get_compile_time_arg_val(6);
    constexpr uint32_t cb_exp = get_compile_time_arg_val(7);
    constexpr uint32_t cb_max = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(9);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(10);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(11);
    constexpr uint32_t cb_out = get_compile_time_arg_val(12);
    constexpr uint32_t users = get_compile_time_arg_val(13);
    constexpr uint32_t cr = get_compile_time_arg_val(14);
    constexpr uint32_t Wt = get_compile_time_arg_val(15);
    constexpr uint32_t W = 2 * cr;
    constexpr uint32_t in_tiles = users * cr * Wt;
    constexpr uint32_t bias_tiles = cr * Wt;
    constexpr uint32_t out_tiles = users * Wt;

    CircularBuffer prev_kv_cb(cb_prev_kv);
    CircularBuffer prev_gate_cb(cb_prev_gate);
    CircularBuffer bias_ca_cb(cb_bias_ca);
    CircularBuffer win_kv_cb(cb_win_kv);
    CircularBuffer win_gate_cb(cb_win_gate);
    CircularBuffer bias_cb_cb(cb_bias_cb);
    CircularBuffer logits_cb(cb_logits);
    CircularBuffer exp_cb(cb_exp);
    CircularBuffer max_cb(cb_max);
    CircularBuffer sum_cb(cb_sum);
    CircularBuffer acc_cb(cb_acc);
    CircularBuffer weight_cb(cb_weight);
    CircularBuffer out_cb(cb_out);

    compute_kernel_hw_startup(cb_prev_gate, cb_bias_ca, cb_out);

    prev_kv_cb.reserve_back(in_tiles);
    prev_kv_cb.push_back(in_tiles);
    prev_gate_cb.reserve_back(in_tiles);
    prev_gate_cb.push_back(in_tiles);
    bias_ca_cb.reserve_back(bias_tiles);
    bias_ca_cb.push_back(bias_tiles);

    prev_kv_cb.wait_front(in_tiles);
    prev_gate_cb.wait_front(in_tiles);
    bias_ca_cb.wait_front(bias_tiles);
    win_kv_cb.wait_front(in_tiles);
    win_gate_cb.wait_front(in_tiles);
    bias_cb_cb.wait_front(bias_tiles);

    out_cb.reserve_back(out_tiles);

    for (uint32_t u = 0; u < users; ++u) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            logits_cb.reserve_back(W);
            for (uint32_t t = 0; t < cr; ++t) {
                const uint32_t in_idx = (u * cr + t) * Wt + wt;
                const uint32_t bias_idx = t * Wt + wt;
                add_faces(cb_prev_gate, in_idx, cb_bias_ca, bias_idx, cb_logits, t);
                add_faces(cb_win_gate, in_idx, cb_bias_cb, bias_idx, cb_logits, cr + t);
            }
            logits_cb.push_back(W);
            logits_cb.wait_front(W);

            max_cb.reserve_back(1);
            copy_face(cb_logits, 0, cb_max);
            max_cb.push_back(1);
            for (uint32_t i = 1; i < W; ++i) {
                max_cb.wait_front(1);
                copy_tile_init_with_dt(cb_max);
                tile_regs_acquire();
                copy_tile(cb_max, 0, 0);
                copy_tile_init_with_dt(cb_logits);
                copy_tile(cb_logits, i, 1);
                binary_max_tile_init();
                binary_max_tile(0, 1, 0, VectorMode::R);
                tile_regs_commit();
                tile_regs_wait();
                max_cb.pop_front(1);
                max_cb.reserve_back(1);
                pack_face(cb_max);
                tile_regs_release();
                max_cb.push_back(1);
            }
            max_cb.wait_front(1);

            exp_cb.reserve_back(W);
            sub_init(cb_logits, cb_max);
            exp_tile_init();
            for (uint32_t i = 0; i < W; ++i) {
                tile_regs_acquire();
                sub_tiles(cb_logits, cb_max, i, 0, 0);
                exp_tile(0, VectorMode::R);
                tile_regs_commit();
                tile_regs_wait();
                pack_face(cb_exp, i);
                tile_regs_release();
            }
            exp_cb.push_back(W);
            exp_cb.wait_front(W);

            sum_cb.reserve_back(1);
            copy_face(cb_exp, 0, cb_sum);
            sum_cb.push_back(1);
            for (uint32_t i = 1; i < W; ++i) {
                sum_cb.wait_front(1);
                add_init(cb_sum, cb_exp);
                tile_regs_acquire();
                add_tiles(cb_sum, cb_exp, 0, i, 0);
                tile_regs_commit();
                tile_regs_wait();
                sum_cb.pop_front(1);
                sum_cb.reserve_back(1);
                pack_face(cb_sum);
                tile_regs_release();
                sum_cb.push_back(1);
            }
            sum_cb.wait_front(1);

            copy_tile_init_with_dt(cb_sum);
            tile_regs_acquire();
            copy_tile(cb_sum, 0, 0);
            recip_tile_init();
            recip_tile(0, VectorMode::R);
            tile_regs_commit();
            tile_regs_wait();
            sum_cb.pop_front(1);
            sum_cb.reserve_back(1);
            pack_face(cb_sum);
            tile_regs_release();
            sum_cb.push_back(1);
            sum_cb.wait_front(1);
            for (uint32_t i = 0; i < W; ++i) {
                const uint32_t kv_cb = (i < cr) ? cb_prev_kv : cb_win_kv;
                const uint32_t tok = (i < cr) ? i : (i - cr);
                const uint32_t kv_idx = (u * cr + tok) * Wt + wt;

                weight_cb.reserve_back(1);
                mul_init(cb_exp, cb_sum);
                tile_regs_acquire();
                mul_tiles(cb_exp, cb_sum, i, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_face(cb_weight);
                tile_regs_release();
                weight_cb.push_back(1);
                weight_cb.wait_front(1);

                mul_init(cb_weight, kv_cb);
                tile_regs_acquire();
                mul_tiles(cb_weight, kv_cb, 0, kv_idx, 0);
                tile_regs_commit();
                tile_regs_wait();
                weight_cb.pop_front(1);

                if (i == 0) {
                    acc_cb.reserve_back(1);
                    pack_face(cb_acc);
                    tile_regs_release();
                    acc_cb.push_back(1);
                } else {
                    weight_cb.reserve_back(1);
                    pack_face(cb_weight);
                    tile_regs_release();
                    weight_cb.push_back(1);
                    weight_cb.wait_front(1);
                    acc_cb.wait_front(1);
                    add_init(cb_acc, cb_weight);
                    tile_regs_acquire();
                    add_tiles(cb_acc, cb_weight, 0, 0, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    acc_cb.pop_front(1);
                    weight_cb.pop_front(1);
                    acc_cb.reserve_back(1);
                    pack_face(cb_acc);
                    tile_regs_release();
                    acc_cb.push_back(1);
                }
            }

            acc_cb.wait_front(1);
            copy_tile_init_with_dt(cb_acc);
            tile_regs_acquire();
            copy_tile(cb_acc, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_face(cb_out, u * Wt + wt);
            tile_regs_release();
            acc_cb.pop_front(1);
            max_cb.pop_front(1);
            sum_cb.pop_front(1);
            exp_cb.pop_front(W);
            logits_cb.pop_front(W);
        }
    }

    out_cb.push_back(out_tiles);
}
