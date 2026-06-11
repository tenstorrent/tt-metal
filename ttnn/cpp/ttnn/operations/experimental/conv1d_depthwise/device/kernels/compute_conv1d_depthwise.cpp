// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// Accumulate one tap-block into out_cb: out[i] = tilized[i] * scalar (+ prior partial when tap>0).
// `scalar_cb` holds a single resident tile (= taps[tap]) reused for every tile in the block.
template <uint32_t block_num_tiles>
inline void mul_and_accumulate(uint32_t tilized_cb, uint32_t scalar_cb, uint32_t out_cb, uint32_t tap) {
    cb_wait_front(tilized_cb, block_num_tiles);
    // scalar_cb holds the K resident tap tiles (filled once); read tile `tap`, never popped.

    for (uint32_t i = 0; i < block_num_tiles; ++i) {
        tile_regs_acquire();
        reconfig_data_format_srca(tilized_cb);
        reconfig_data_format_srcb(scalar_cb);
        mul_tiles_init(tilized_cb, scalar_cb);
        mul_tiles(tilized_cb, scalar_cb, i, tap, 0);

        if (tap != 0) {
            reconfig_data_format_srca(out_cb);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(out_cb);
            cb_wait_front(out_cb, 1);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(out_cb, 0, 0);
            cb_pop_front(out_cb, 1);
        }
        tile_regs_commit();

        cb_reserve_back(out_cb, 1);
        tile_regs_wait();
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        tile_regs_release();
    }

    cb_pop_front(tilized_cb, block_num_tiles);
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

    // The K tap tiles are filled once by the reader and stay resident — wait for them once.
    cb_wait_front(scalar_cb, K);

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        for (uint32_t tap = 0; tap < K; ++tap) {
            // Lossless: this is the fp32 numerical baseline (replaces an all-fp32 MAC path);
            // the default Fast mode truncates fp32→tf32 during tilize.
            compute_kernel_lib::tilize<
                block_w_tiles,
                act_cb,
                tilized_cb,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
                compute_kernel_lib::tilize_config::Fp32Mode::Lossless>(block_h_tiles);
            mul_and_accumulate<block_num_tiles>(tilized_cb, scalar_cb, out_cb, tap);
        }
        // out_cb now holds the accumulated tiled block; untilize to row-major for the writer.
        compute_kernel_lib::untilize<block_w_tiles, out_cb, out_rm_cb>(block_h_tiles);
    }
}
