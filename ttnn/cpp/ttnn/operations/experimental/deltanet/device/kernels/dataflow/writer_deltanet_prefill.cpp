// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer kernel for DeltaNet prefill (S>1 token loop).
//
// Per-token: writes Dv_tiles output tiles to DRAM.
// After loop: writes final recurrent state and conv_state.
//
// Output tensor shape: [S*H, 1, 1, Dv]
//   Token s, head h → tile row (s * H + h), so start tile = (s * H + head_idx) * Dv_tiles
//
// State ping-pong: cb_state_A/B are exclusively compute-produced.
//   s=0 writes to A, s=1 to B, s=2 to A, ...
//   Final state: S odd → cb_state_A, S even → cb_state_B.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_state_A        = get_compile_time_arg_val(0);
    constexpr uint32_t cb_state_B        = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output         = get_compile_time_arg_val(2);
    constexpr uint32_t Dk_tiles          = get_compile_time_arg_val(3);
    constexpr uint32_t Dv_tiles          = get_compile_time_arg_val(4);
    constexpr uint32_t cb_conv_state_out = get_compile_time_arg_val(5);
    constexpr uint32_t head_expand       = get_compile_time_arg_val(6);
    constexpr uint32_t S                 = get_compile_time_arg_val(7);
    constexpr uint32_t H                 = get_compile_time_arg_val(8);
    constexpr auto accessor_args         = TensorAccessorArgs<9>();

    const uint32_t state_out_addr       = get_arg_val<uint32_t>(0);
    const uint32_t output_addr          = get_arg_val<uint32_t>(1);
    const uint32_t state_out_start_tile = get_arg_val<uint32_t>(2);
    const uint32_t conv_state_out_addr  = get_arg_val<uint32_t>(3);
    const uint32_t head_idx             = get_arg_val<uint32_t>(4);
    const uint32_t conv_q_tile          = get_arg_val<uint32_t>(5);
    const uint32_t conv_k_tile          = get_arg_val<uint32_t>(6);
    const uint32_t conv_v_tile          = get_arg_val<uint32_t>(7);

    constexpr uint32_t state_tiles = Dk_tiles * Dv_tiles;
    constexpr uint32_t TILES_PER_COMPONENT = 4;

    const uint32_t tile_bytes_state = get_tile_size(cb_state_A);
    const uint32_t tile_bytes_out   = get_tile_size(cb_output);
    const uint32_t tile_bytes_conv  = get_tile_size(cb_conv_state_out);

    const auto state_out_acc = TensorAccessor(accessor_args, state_out_addr, tile_bytes_state);
    const auto output_acc    = TensorAccessor(accessor_args, output_addr, tile_bytes_out);
    const auto conv_out_acc  = TensorAccessor(accessor_args, conv_state_out_addr, tile_bytes_conv);

    for (uint32_t s = 0; s < S; s++) {
        cb_wait_front(cb_output, Dv_tiles);
        uint32_t l1_addr = get_read_ptr(cb_output);
        uint32_t out_start = (s * H + head_idx) * Dv_tiles;
        for (uint32_t t = 0; t < Dv_tiles; t++) {
            noc_async_write_tile(out_start + t, output_acc, l1_addr);
            l1_addr += tile_bytes_out;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_output, Dv_tiles);
    }

    // Write final recurrent state
    {
        constexpr uint32_t final_state_cb = (S % 2 == 1) ? cb_state_A : cb_state_B;
        cb_wait_front(final_state_cb, state_tiles);
        uint32_t l1_addr = get_read_ptr(final_state_cb);
        for (uint32_t t = 0; t < state_tiles; t++) {
            noc_async_write_tile(state_out_start_tile + t, state_out_acc, l1_addr);
            l1_addr += tile_bytes_state;
        }
        noc_async_write_barrier();
        cb_pop_front(final_state_cb, state_tiles);
    }

    // Write updated conv_state tiles
    {
        bool is_first_in_group = (head_idx % head_expand == 0);

        cb_wait_front(cb_conv_state_out, 12);
        uint32_t l1_base = get_read_ptr(cb_conv_state_out);

        if (is_first_in_group) {
            uint32_t l1_addr = l1_base;
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                noc_async_write_tile(conv_q_tile + t, conv_out_acc, l1_addr);
                l1_addr += tile_bytes_conv;
            }
            for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
                noc_async_write_tile(conv_k_tile + t, conv_out_acc, l1_addr);
                l1_addr += tile_bytes_conv;
            }
        }

        uint32_t v_l1 = l1_base + 8 * tile_bytes_conv;
        for (uint32_t t = 0; t < TILES_PER_COMPONENT; t++) {
            noc_async_write_tile(conv_v_tile + t, conv_out_acc, v_l1);
            v_l1 += tile_bytes_conv;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_conv_state_out, 12);
    }
}
