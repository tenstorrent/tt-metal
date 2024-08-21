// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

constexpr uint32_t NUM_TILES_IN_TILIZED_CHUNK = 32;
constexpr uint32_t NUM_BYTES_IN_BFLOAT16 = 2;
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t NUM_BYTES_IN_TILIZED_CHUNK = NUM_TILES_IN_TILIZED_CHUNK * TILE_WIDTH * NUM_BYTES_IN_BFLOAT16;

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    const uint32_t hidden_state_len = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_h_acc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h_out = get_compile_time_arg_val(2);

    const uint32_t hidden_state_len_bytes = hidden_state_len * NUM_BYTES_IN_BFLOAT16;

    cb_wait_front(cb_out, num_tiles_per_core);

    // This assumes we are updating h input tensor in-place
    uint32_t src_addr = get_read_ptr(cb_h_acc);
    uint64_t dst_addr = get_noc_addr(get_write_ptr(cb_h_out));
    noc_async_write(src_addr, dst_addr, hidden_state_len_bytes);
    noc_async_write_barrier();
}
