// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DeltaNet recurrence reader kernel — minimal passthrough version
//
// Reads state, q, k, v tensors from DRAM into circular buffers.
// Pushes dummy decay and beta tiles for the passthrough compute kernel.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    uint32_t state_addr = get_arg_val<uint32_t>(0);
    uint32_t conv_out_addr = get_arg_val<uint32_t>(1);

    // Compile-time args
    constexpr uint32_t num_heads = get_compile_time_arg_val(0);   // 48
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);     // 4
    constexpr uint32_t v_tiles = get_compile_time_arg_val(2);     // 4
    constexpr uint32_t state_tiles_per_head = k_tiles * v_tiles;  // 16

    // TensorAccessor args start at compile-time arg index 3
    constexpr auto state_acc_args = TensorAccessorArgs<3>();
    constexpr auto conv_out_acc_args = TensorAccessorArgs<state_acc_args.next_compile_time_args_offset()>();

    // CB indices
    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;
    constexpr uint32_t cb_decay = tt::CBIndex::c_3;
    constexpr uint32_t cb_beta = tt::CBIndex::c_4;
    constexpr uint32_t cb_state = tt::CBIndex::c_5;

    uint32_t state_tile_bytes = get_tile_size(cb_state);
    uint32_t conv_tile_bytes = get_tile_size(cb_q);

    const auto s_state = TensorAccessor(state_acc_args, state_addr, state_tile_bytes);
    const auto s_conv = TensorAccessor(conv_out_acc_args, conv_out_addr, conv_tile_bytes);

    for (uint32_t head = 0; head < num_heads; head++) {
        // Read state tiles for this head
        uint32_t state_start_tile = head * state_tiles_per_head;
        for (uint32_t st = 0; st < state_tiles_per_head; st++) {
            cb_reserve_back(cb_state, 1);
            uint32_t l1_addr = get_write_ptr(cb_state);
            noc_async_read_tile(state_start_tile + st, s_state, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_state, 1);
        }

        // Push dummy decay tiles (state_tiles_per_head)
        for (uint32_t dt = 0; dt < state_tiles_per_head; dt++) {
            cb_reserve_back(cb_decay, 1);
            cb_push_back(cb_decay, 1);
        }

        // Read Q tiles from conv_out (passthrough: just read valid tiles)
        for (uint32_t qt = 0; qt < k_tiles; qt++) {
            cb_reserve_back(cb_q, 1);
            uint32_t l1_addr = get_write_ptr(cb_q);
            noc_async_read_tile(qt, s_conv, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_q, 1);
        }

        // Read K tiles from conv_out (passthrough: just read valid tiles)
        for (uint32_t kt = 0; kt < k_tiles; kt++) {
            cb_reserve_back(cb_k, 1);
            uint32_t l1_addr = get_write_ptr(cb_k);
            noc_async_read_tile(kt, s_conv, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_k, 1);
        }

        // Read V tiles from conv_out (passthrough: just read valid tiles)
        for (uint32_t vt = 0; vt < v_tiles; vt++) {
            cb_reserve_back(cb_v, 1);
            uint32_t l1_addr = get_write_ptr(cb_v);
            noc_async_read_tile(vt, s_conv, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_v, 1);
        }

        // Push dummy beta tiles (v_tiles)
        for (uint32_t bt = 0; bt < v_tiles; bt++) {
            cb_reserve_back(cb_beta, 1);
            cb_push_back(cb_beta, 1);
        }
    }
}
