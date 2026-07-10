// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer kernel for fully fused DeltaNet decode (phase B2).
//
// Writes per-head:
//   - Updated recurrent state: Dk_tiles * Dv_tiles tiles
//   - Output vector: Dv_tiles tiles
//   - Updated conv_state tiles: v always (4), q+k if first in head group (8 more)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_state_out     = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output        = get_compile_time_arg_val(1);
    constexpr uint32_t Dk_tiles         = get_compile_time_arg_val(2);
    constexpr uint32_t Dv_tiles         = get_compile_time_arg_val(3);
    constexpr uint32_t cb_conv_state_out = get_compile_time_arg_val(4);
    constexpr uint32_t head_expand      = get_compile_time_arg_val(5);
    constexpr auto accessor_args        = TensorAccessorArgs<6>();

    const uint32_t state_out_addr       = get_arg_val<uint32_t>(0);
    const uint32_t output_addr          = get_arg_val<uint32_t>(1);
    const uint32_t state_out_start_tile = get_arg_val<uint32_t>(2);
    const uint32_t output_start_tile    = get_arg_val<uint32_t>(3);
    const uint32_t conv_state_out_addr  = get_arg_val<uint32_t>(4);
    const uint32_t head_idx             = get_arg_val<uint32_t>(5);
    const uint32_t conv_q_tile          = get_arg_val<uint32_t>(6);
    const uint32_t conv_k_tile          = get_arg_val<uint32_t>(7);
    const uint32_t conv_v_tile          = get_arg_val<uint32_t>(8);

    constexpr uint32_t state_tiles = Dk_tiles * Dv_tiles;
    constexpr uint32_t TILES_PER_COMPONENT = 4;
    const uint32_t tile_bytes_state = get_tile_size(cb_state_out);
    const uint32_t tile_bytes_out = get_tile_size(cb_output);
    const uint32_t tile_bytes_conv = get_tile_size(cb_conv_state_out);

    const auto state_out_acc = TensorAccessor(accessor_args, state_out_addr, tile_bytes_state);
    const auto output_acc    = TensorAccessor(accessor_args, output_addr, tile_bytes_out);
    const auto conv_out_acc  = TensorAccessor(accessor_args, conv_state_out_addr, tile_bytes_conv);

    // Write updated recurrent state
    {
        cb_wait_front(cb_state_out, state_tiles);
        uint32_t l1_addr = get_read_ptr(cb_state_out);
        for (uint32_t t = 0; t < state_tiles; t++) {
            noc_async_write_tile(state_out_start_tile + t, state_out_acc, l1_addr);
            l1_addr += tile_bytes_state;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_state_out, state_tiles);
    }

    // Write output vector
    {
        cb_wait_front(cb_output, Dv_tiles);
        uint32_t l1_addr = get_read_ptr(cb_output);
        for (uint32_t t = 0; t < Dv_tiles; t++) {
            noc_async_write_tile(output_start_tile + t, output_acc, l1_addr);
            l1_addr += tile_bytes_out;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_output, Dv_tiles);
    }

    // Write updated conv_state tiles
    // CB layout: [q_tiles(4), k_tiles(4), v_tiles(4)]
    // All cores write their v tiles (unique per head).
    // Only first core in each head group writes q+k tiles (shared among group).
    {
        bool is_first_in_group = (head_idx % head_expand == 0);
        uint32_t total_conv_tiles = is_first_in_group ? 12 : 4;
        uint32_t cb_offset_tiles = is_first_in_group ? 0 : 8;

        cb_wait_front(cb_conv_state_out, 12);
        uint32_t l1_addr = get_read_ptr(cb_conv_state_out) + cb_offset_tiles * tile_bytes_conv;

        if (is_first_in_group) {
            // Write q tiles
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                noc_async_write_tile(conv_q_tile + t, conv_out_acc, l1_addr);
                l1_addr += tile_bytes_conv;
            }
            // Write k tiles
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                noc_async_write_tile(conv_k_tile + t, conv_out_acc, l1_addr);
                l1_addr += tile_bytes_conv;
            }
        }

        // Write v tiles (all cores)
        uint32_t v_l1 = get_read_ptr(cb_conv_state_out) + 8 * tile_bytes_conv;
        for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
            noc_async_write_tile(conv_v_tile + t, conv_out_acc, v_l1);
            v_l1 += tile_bytes_conv;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_conv_state_out, 12);
    }
}
