// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/conv/conv1d/conv1d_depthwise_helpers.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// Compute one block (one kernel-tap slice) of a 1D depthwise conv. idx is the kernel-tap block
// index (0 .. filter_h*filter_w-1); idx==0 seeds out_cb with the tap-0 product, later taps
// accumulate via dest-reuse. See depthwise_fir_mac_tile for the per-tile mul/add/pack body.
inline void mul_and_accumulate_block(
    experimental::CB in0_cb, experimental::CB in1_cb, experimental::CB out_cb, uint32_t block_num_tiles, uint32_t idx) {
    const uint32_t in0_cb_id = in0_cb.get_cb_id();
    const uint32_t in1_cb_id = in1_cb.get_cb_id();
    const uint32_t out_cb_id = out_cb.get_cb_id();

    for (uint32_t i = 0; i < block_num_tiles; i++) {
        in1_cb.wait_front(1);
        in0_cb.wait_front(1);

        // Per-channel weight tiles streamed through in1 (one per output tile, index 0).
        conv1d_depthwise::depthwise_fir_mac_tile(in0_cb_id, 0, in1_cb_id, 0, out_cb_id, idx == 0);

        in0_cb.pop_front(1);
        in1_cb.pop_front(1);
    }
}

template <uint32_t in0_block_w, uint32_t kernel_width, uint32_t block_num_tiles>
inline void mul_and_accumulate_coalesced_block(
    experimental::CB in0_cb, experimental::CB in1_cb, experimental::CB out_cb) {
    static_assert(kernel_width > 1);
    static_assert(in0_block_w % kernel_width == 0);
    static_assert(block_num_tiles % in0_block_w == 0);

    constexpr uint32_t in_channels_ntiles = in0_block_w / kernel_width;
    constexpr uint32_t act_block_h_ntiles = block_num_tiles / in0_block_w;

    const uint32_t in0_cb_id = in0_cb.get_cb_id();
    const uint32_t in1_cb_id = in1_cb.get_cb_id();
    const uint32_t out_cb_id = out_cb.get_cb_id();

    in0_cb.wait_front(block_num_tiles);
    in1_cb.wait_front(block_num_tiles);

    for (uint32_t h = 0; h < act_block_h_ntiles; ++h) {
        for (uint32_t c = 0; c < in_channels_ntiles; ++c) {
            tile_regs_acquire();
            reconfig_data_format_srca(in0_cb_id);
            reconfig_data_format_srcb(in1_cb_id);

            for (uint32_t tap = 0; tap < kernel_width; ++tap) {
                const uint32_t act_tile_idx = h * in0_block_w + tap * in_channels_ntiles + c;
                const uint32_t weight_tile_idx =
                    tap * act_block_h_ntiles * in_channels_ntiles + h * in_channels_ntiles + c;
                mul_tiles_init(in0_cb_id, in1_cb_id, tap != 0 ? 1U : 0U, __builtin_LINE());
                mul_tiles(in0_cb_id, in1_cb_id, act_tile_idx, weight_tile_idx, 0);
            }
            tile_regs_commit();

            out_cb.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, out_cb_id);
            out_cb.push_back(1);
            tile_regs_release();
        }
    }

    in0_cb.pop_front(block_num_tiles);
    in1_cb.pop_front(block_num_tiles);
}

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_num_blocks_h = get_compile_time_arg_val(3);
    constexpr uint32_t in0_num_blocks_w = get_compile_time_arg_val(4);
    constexpr uint32_t in0_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t in1_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t kernel_width = get_compile_time_arg_val(9);
    constexpr bool coalesce_kw_reads = get_compile_time_arg_val(10) == 1;

    experimental::CB cb_tilized_in0(tilized_in0_cb_id);
    experimental::CB cb_in1(in1_cb_id);
    experimental::CB cb_out(out_cb_id);

    // Configures pack for out_cb and math/unpack for in0/in1; pack target never changes after this.
    binary_op_init_common(in0_cb_id, in1_cb_id, out_cb_id);

    for (uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
        for (uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
            compute_kernel_lib::tilize<in0_block_w, in0_cb_id, tilized_in0_cb_id>(in0_num_subblocks);
            reconfig_data_format_srca(tilized_in0_cb_id);
            if constexpr (coalesce_kw_reads) {
                mul_and_accumulate_coalesced_block<in0_block_w, kernel_width, in0_block_num_tiles>(
                    cb_tilized_in0, cb_in1, cb_out);
            } else {
                const uint32_t idx = in0_block_h_i * in0_num_blocks_w + in0_block_w_i;
                mul_and_accumulate_block(cb_tilized_in0, cb_in1, cb_out, in0_block_num_tiles, idx);
            }
        }
    }
}
