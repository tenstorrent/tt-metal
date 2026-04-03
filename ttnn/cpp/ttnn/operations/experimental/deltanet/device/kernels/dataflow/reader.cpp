// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DeltaNet recurrence reader — interleaved state+decay for double-buffered CBs
//
// Push order per head (matching compute's consumption):
//   1. state[0], decay[0], state[1], decay[1], ... (interleaved, 16 pairs)
//   2. Q tiles (k_tiles)
//   3. K tiles (k_tiles)
//   4. V tiles (v_tiles)
//   5. beta tiles (v_tiles dummy)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t state_addr = get_arg_val<uint32_t>(0);
    uint32_t conv_out_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_heads = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t v_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t state_tiles_per_head = k_tiles * v_tiles;

    constexpr auto state_acc_args = TensorAccessorArgs<3>();
    constexpr auto conv_out_acc_args = TensorAccessorArgs<state_acc_args.next_compile_time_args_offset()>();

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
        uint32_t state_start_tile = head * state_tiles_per_head;

        // INTERLEAVED: push state[i] then decay[i] for each i
        // This matches compute's consumption: wait(state,1), wait(decay,1) per iteration
        for (uint32_t st = 0; st < state_tiles_per_head; st++) {
            // Push one state tile
            cb_reserve_back(cb_state, 1);
            uint32_t l1_addr = get_write_ptr(cb_state);
            noc_async_read_tile(state_start_tile + st, s_state, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_state, 1);

            // Push one dummy decay tile
            cb_reserve_back(cb_decay, 1);
            cb_push_back(cb_decay, 1);
        }

        // Q tiles from conv_out
        for (uint32_t qt = 0; qt < k_tiles; qt++) {
            cb_reserve_back(cb_q, 1);
            uint32_t l1_addr = get_write_ptr(cb_q);
            noc_async_read_tile(qt, s_conv, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_q, 1);
        }

        // K tiles from conv_out
        for (uint32_t kt = 0; kt < k_tiles; kt++) {
            cb_reserve_back(cb_k, 1);
            uint32_t l1_addr = get_write_ptr(cb_k);
            noc_async_read_tile(kt, s_conv, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_k, 1);
        }

        // V tiles from conv_out
        for (uint32_t vt = 0; vt < v_tiles; vt++) {
            cb_reserve_back(cb_v, 1);
            uint32_t l1_addr = get_write_ptr(cb_v);
            noc_async_read_tile(vt, s_conv, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_v, 1);
        }

        // Dummy beta tiles
        for (uint32_t bt = 0; bt < v_tiles; bt++) {
            cb_reserve_back(cb_beta, 1);
            cb_push_back(cb_beta, 1);
        }
    }
}
