// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = get_compile_time_arg_val(0);
constexpr uint32_t cb_final = get_compile_time_arg_val(1);
constexpr uint32_t cb_ctl = get_compile_time_arg_val(2);
constexpr uint32_t num_chunks = get_compile_time_arg_val(3);
constexpr uint32_t kt = get_compile_time_arg_val(4);
constexpr auto out_args = TensorAccessorArgs<5>();
constexpr auto final_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

// Writes this core's chunk outputs and its tile rows of the final state.
void kernel_main() {
    size_t arg = 0;
    const uint32_t out_address = get_arg_val<uint32_t>(arg++);
    const uint32_t final_address = get_arg_val<uint32_t>(arg++);
    const uint32_t head = get_arg_val<uint32_t>(arg++);
    const uint32_t first_chunk = get_arg_val<uint32_t>(arg++);

    // The reader's control words: this core's chunk count and whether it forms the final state.
    cb_wait_front(cb_ctl, 1);
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_ctl));
    const uint32_t chunks = words[0];
    const uint32_t final = words[1];
    const uint32_t first_row = words[2];
    const uint32_t row_stride = words[3];
    cb_pop_front(cb_ctl, 1);

    const auto out = TensorAccessor(out_args, out_address);
    const uint32_t out_tile = get_tile_size(cb_out);
    for (uint32_t c = first_chunk; c < first_chunk + chunks; ++c) {
        cb_wait_front(cb_out, kt);
        const uint32_t row = (head * num_chunks + c) * kt;
        for (uint32_t j = 0; j < kt; ++j) {
            noc_async_write_page(row + j, out, get_read_ptr(cb_out) + j * out_tile);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, kt);
    }
    if (final) {
        const auto states = TensorAccessor(final_args, final_address);
        const uint32_t tile = get_tile_size(cb_final);
        // Row j of the state: tiles (j, c).
        for (uint32_t j = first_row; j < kt; j += row_stride) {
            cb_wait_front(cb_final, kt);
            for (uint32_t c = 0; c < kt; ++c) {
                noc_async_write_page(head * kt * kt + j * kt + c, states, get_read_ptr(cb_final) + c * tile);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_final, kt);
        }
    }
    noc_async_write_barrier();
}
