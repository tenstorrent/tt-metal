// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/conv/conv1d/conv1d_depthwise_helpers.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// Accumulate one tap-block into out_cb: out[i] = tilized[i] * scalar (+ prior partial when tap>0).
// scalar_cb holds the K resident tap tiles; read tile `tap`, never popped.
template <uint32_t block_num_tiles>
inline void mul_and_accumulate(uint32_t tilized_cb, uint32_t scalar_cb, uint32_t out_cb, uint32_t tap) {
    experimental::CB tilized(tilized_cb);
    tilized.wait_front(block_num_tiles);
    for (uint32_t i = 0; i < block_num_tiles; ++i) {
        conv1d_depthwise::depthwise_fir_mac_tile(tilized_cb, i, scalar_cb, tap, out_cb, tap == 0);
    }
    tilized.pop_front(block_num_tiles);
}

void kernel_main() {
    constexpr uint32_t act_cb = get_compile_time_arg_val(0);
    constexpr uint32_t scalar_cb = get_compile_time_arg_val(1);
    constexpr uint32_t tilized_cb = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb = get_compile_time_arg_val(3);
    constexpr uint32_t out_rm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t block_w_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t block_h_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t K = get_compile_time_arg_val(7);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    constexpr uint32_t block_num_tiles = block_w_tiles * block_h_tiles;

    binary_op_init_common(tilized_cb, scalar_cb, out_cb);

    // The K tap tiles are filled once by the reader and stay resident; wait for them once.
    experimental::CB scalar(scalar_cb);
    scalar.wait_front(K);

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        for (uint32_t tap = 0; tap < K; ++tap) {
            compute_kernel_lib::tilize<block_w_tiles, act_cb, tilized_cb>(block_h_tiles);
            mul_and_accumulate<block_num_tiles>(tilized_cb, scalar_cb, out_cb, tap);
        }
        // out_cb now holds the accumulated tiled block; untilize to row-major for the writer.
        compute_kernel_lib::untilize<block_w_tiles, out_cb, out_rm_cb>(block_h_tiles);
    }
}
