// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {

// out = sigmoid(w * scale + bias) [+ eps | * post_mul], computed for a single [1,H] row tile.
void fused_sigmoid_with_bias_and_scale(
    uint32_t cb_w,
    uint32_t cb_bias,
    uint32_t cb_scratch,
    uint32_t cb_out,
    uint32_t scale_bits,
    uint32_t post_mul_bits,
    bool add_eps,
    uint32_t eps_bits) {
    cb_wait_front(cb_w, 1);
    cb_wait_front(cb_bias, 1);

    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_w);
    copy_tile(cb_w, 0, 0);
    mul_unary_tile(0, scale_bits);
    tile_regs_commit();

    cb_reserve_back(cb_scratch, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_scratch);
    pack_tile(0, cb_scratch);
    tile_regs_release();
    cb_push_back(cb_scratch, 1);
    cb_pop_front(cb_w, 1);

    cb_wait_front(cb_scratch, 1);

    tile_regs_acquire();
    add_bcast_rows_init_short(cb_scratch, cb_bias);
    add_tiles_bcast<BroadcastType::ROW>(cb_scratch, cb_bias, 0, 0, 0);
    sigmoid_tile_init();
    sigmoid_tile(0);
    if (add_eps) {
        add_unary_tile(0, eps_bits);
    } else if (post_mul_bits != 0) {
        mul_unary_tile(0, post_mul_bits);
    }
    tile_regs_commit();

    cb_pop_front(cb_scratch, 1);
    cb_pop_front(cb_bias, 1);

    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t d_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_pre_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_post_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_pre_bias = get_compile_time_arg_val(2);
    constexpr uint32_t cb_post_bias = get_compile_time_arg_val(3);
    constexpr uint32_t cb_hidden = get_compile_time_arg_val(4);
    constexpr uint32_t cb_post_out = get_compile_time_arg_val(5);
    constexpr uint32_t cb_collapsed = get_compile_time_arg_val(6);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(7);
    constexpr uint32_t cb_pre = get_compile_time_arg_val(8);
    constexpr uint32_t pre_scale_bits = get_compile_time_arg_val(9);
    constexpr uint32_t post_scale_bits = get_compile_time_arg_val(10);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(11);
    constexpr uint32_t two_bits = get_compile_time_arg_val(12);

    binary_op_init_common(cb_pre_w, cb_pre_bias, cb_pre);

    // pre  = sigmoid(pre_w  * pre_scale  + pre_bias) + eps   -> cb_pre   (matmul in0).
    // post = 2 * sigmoid(post_w * post_scale + post_bias)    -> cb_post_out.
    fused_sigmoid_with_bias_and_scale(cb_pre_w, cb_pre_bias, cb_scratch, cb_pre, pre_scale_bits, 0, true, eps_bits);
    fused_sigmoid_with_bias_and_scale(
        cb_post_w, cb_post_bias, cb_scratch, cb_post_out, post_scale_bits, two_bits, false, eps_bits);

    // collapsed = pre[1,H] @ hidden[H,D] -> [1,D]. Padding rows of hidden (>= H) are zero, so
    // the K reduction only accumulates the H valid streams regardless of pre's padding values.
    mm_init(cb_pre, cb_hidden, cb_collapsed);
    cb_wait_front(cb_pre, 1);
    cb_wait_front(cb_hidden, d_tiles);
    for (uint32_t n = 0; n < d_tiles; ++n) {
        tile_regs_acquire();
        matmul_tiles(cb_pre, cb_hidden, 0, n, 0);
        tile_regs_commit();

        cb_reserve_back(cb_collapsed, 1);
        tile_regs_wait();
        pack_tile(0, cb_collapsed);
        tile_regs_release();
        cb_push_back(cb_collapsed, 1);
    }
    cb_pop_front(cb_pre, 1);
    cb_pop_front(cb_hidden, d_tiles);
}
